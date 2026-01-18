from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from thinc.api import Config, Model, Optimizer, set_dropout_rate
from ..errors import Errors
from ..language import Language
from ..tokens import Doc
from ..training import Example, validate_examples, validate_get_examples
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
class Tok2Vec(TrainablePipe):
    """Apply a "token-to-vector" model and set its outputs in the doc.tensor
    attribute. This is mostly useful to share a single subnetwork between multiple
    components, e.g. to have one embedding and CNN network shared between a
    parser, tagger and NER.

    In order to use the `Tok2Vec` predictions, subsequent components should use
    the `Tok2VecListener` layer as the tok2vec subnetwork of their model. This
    layer will read data from the `doc.tensor` attribute during prediction.
    During training, the `Tok2Vec` component will save its prediction and backprop
    callback for each batch, so that the subsequent components can backpropagate
    to the shared weights. This implementation is used because it allows us to
    avoid relying on object identity within the models to achieve the parameter
    sharing.
    """

    def __init__(self, vocab: Vocab, model: Model, name: str='tok2vec') -> None:
        """Initialize a tok2vec component.

        vocab (Vocab): The shared vocabulary.
        model (thinc.api.Model[List[Doc], List[Floats2d]]):
            The Thinc Model powering the pipeline component. It should take
            a list of Doc objects as input, and output a list of 2d float arrays.
        name (str): The component instance name.

        DOCS: https://spacy.io/api/tok2vec#init
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self.listener_map: Dict[str, List['Tok2VecListener']] = {}
        self.cfg: Dict[str, Any] = {}

    @property
    def listeners(self) -> List['Tok2VecListener']:
        """RETURNS (List[Tok2VecListener]): The listener models listening to this
        component. Usually internals.
        """
        return [m for c in self.listening_components for m in self.listener_map[c]]

    @property
    def listening_components(self) -> List[str]:
        """RETURNS (List[str]): The downstream components listening to this
        component. Usually internals.
        """
        return list(self.listener_map.keys())

    def add_listener(self, listener: 'Tok2VecListener', component_name: str) -> None:
        """Add a listener for a downstream component. Usually internals."""
        self.listener_map.setdefault(component_name, [])
        if listener not in self.listener_map[component_name]:
            self.listener_map[component_name].append(listener)

    def remove_listener(self, listener: 'Tok2VecListener', component_name: str) -> bool:
        """Remove a listener for a downstream component. Usually internals."""
        if component_name in self.listener_map:
            if listener in self.listener_map[component_name]:
                self.listener_map[component_name].remove(listener)
                if not self.listener_map[component_name]:
                    del self.listener_map[component_name]
                return True
        return False

    def find_listeners(self, component) -> None:
        """Walk over a model of a processing component, looking for layers that
        are Tok2vecListener subclasses that have an upstream_name that matches
        this component. Listeners can also set their upstream_name attribute to
        the wildcard string '*' to match any `Tok2Vec`.

        You're unlikely to ever need multiple `Tok2Vec` components, so it's
        fine to leave your listeners upstream_name on '*'.
        """
        names = ('*', self.name)
        if isinstance(getattr(component, 'model', None), Model):
            for node in component.model.walk():
                if isinstance(node, Tok2VecListener) and node.upstream_name in names:
                    self.add_listener(node, component.name)

    def predict(self, docs: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns a single tensor for a batch of documents.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS: Vector representations for each token in the documents.

        DOCS: https://spacy.io/api/tok2vec#predict
        """
        if not any((len(doc) for doc in docs)):
            width = self.model.get_dim('nO')
            return [self.model.ops.alloc((0, width)) for doc in docs]
        tokvecs = self.model.predict(docs)
        return tokvecs

    def set_annotations(self, docs: Sequence[Doc], tokvecses) -> None:
        """Modify a batch of documents, using pre-computed scores.

        docs (Iterable[Doc]): The documents to modify.
        tokvecses: The tensors to set, produced by Tok2Vec.predict.

        DOCS: https://spacy.io/api/tok2vec#set_annotations
        """
        for doc, tokvecs in zip(docs, tokvecses):
            assert tokvecs.shape[0] == len(doc)
            doc.tensor = tokvecs

    def update(self, examples: Iterable[Example], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None):
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model.

        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.

        DOCS: https://spacy.io/api/tok2vec#update
        """
        if losses is None:
            losses = {}
        validate_examples(examples, 'Tok2Vec.update')
        docs = [eg.predicted for eg in examples]
        set_dropout_rate(self.model, drop)
        tokvecs, bp_tokvecs = self.model.begin_update(docs)
        d_tokvecs = [self.model.ops.alloc2f(*t2v.shape) for t2v in tokvecs]
        losses.setdefault(self.name, 0.0)

        def accumulate_gradient(one_d_tokvecs):
            """Accumulate tok2vec loss and gradient. This is passed as a callback
            to all but the last listener. Only the last one does the backprop.
            """
            nonlocal d_tokvecs
            for i in range(len(one_d_tokvecs)):
                d_tokvecs[i] += one_d_tokvecs[i]
                losses[self.name] += float((one_d_tokvecs[i] ** 2).sum())
            return [self.model.ops.alloc2f(*t2v.shape) for t2v in tokvecs]

        def backprop(one_d_tokvecs):
            """Callback to actually do the backprop. Passed to last listener."""
            accumulate_gradient(one_d_tokvecs)
            d_docs = bp_tokvecs(d_tokvecs)
            if sgd is not None:
                self.finish_update(sgd)
            return d_docs
        batch_id = Tok2VecListener.get_batch_id(docs)
        for listener in self.listeners[:-1]:
            listener.receive(batch_id, tokvecs, accumulate_gradient)
        if self.listeners:
            self.listeners[-1].receive(batch_id, tokvecs, backprop)
        return losses

    def get_loss(self, examples, scores) -> None:
        pass

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Optional[Language]=None):
        """Initialize the pipe for training, using a representative set
        of data examples.

        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.

        DOCS: https://spacy.io/api/tok2vec#initialize
        """
        validate_get_examples(get_examples, 'Tok2Vec.initialize')
        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
        assert doc_sample, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample)

    def add_label(self, label):
        raise NotImplementedError