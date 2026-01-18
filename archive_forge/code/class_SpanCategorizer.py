from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import numpy
from thinc.api import Config, Model, Ops, Optimizer, get_current_ops, set_dropout_rate
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ..compat import Protocol, runtime_checkable
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span, SpanGroup
from ..training import Example, validate_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
class SpanCategorizer(TrainablePipe):
    """Pipeline component to label spans of text.

    DOCS: https://spacy.io/api/spancategorizer
    """

    def __init__(self, vocab: Vocab, model: Model[Tuple[List[Doc], Ragged], Floats2d], suggester: Suggester, name: str='spancat', *, add_negative_label: bool=False, spans_key: str='spans', negative_weight: Optional[float]=1.0, allow_overlap: Optional[bool]=True, max_positive: Optional[int]=None, threshold: Optional[float]=0.5, scorer: Optional[Callable]=spancat_score) -> None:
        """Initialize the multi-label or multi-class span categorizer.

        vocab (Vocab): The shared vocabulary.
        model (thinc.api.Model): The Thinc Model powering the pipeline component.
            For multi-class classification (single label per span) we recommend
            using a Softmax classifier as a the final layer, while for multi-label
            classification (multiple possible labels per span) we recommend Logistic.
        suggester (Callable[[Iterable[Doc], Optional[Ops]], Ragged]): A function that suggests spans.
            Spans are returned as a ragged array with two integer columns, for the
            start and end positions.
        name (str): The component instance name, used to add entries to the
            losses during training.
        spans_key (str): Key of the Doc.spans dict to save the spans under.
            During initialization and training, the component will look for
            spans on the reference document under the same key. Defaults to
            `"spans"`.
        add_negative_label (bool): Learn to predict a special 'negative_label'
            when a Span is not annotated.
        threshold (Optional[float]): Minimum probability to consider a prediction
            positive. Defaults to 0.5. Spans with a positive prediction will be saved
            on the Doc.
        max_positive (Optional[int]): Maximum number of labels to consider
            positive per span. Defaults to None, indicating no limit.
        negative_weight (float): Multiplier for the loss terms.
            Can be used to downweight the negative samples if there are too many
            when add_negative_label is True. Otherwise its unused.
        allow_overlap (bool): If True the data is assumed to contain overlapping spans.
            Otherwise it produces non-overlapping spans greedily prioritizing
            higher assigned label scores. Only used when max_positive is 1.
        scorer (Optional[Callable]): The scoring method. Defaults to
            Scorer.score_spans for the Doc.spans[spans_key] with overlapping
            spans allowed.

        DOCS: https://spacy.io/api/spancategorizer#init
        """
        self.cfg = {'labels': [], 'spans_key': spans_key, 'threshold': threshold, 'max_positive': max_positive, 'negative_weight': negative_weight, 'allow_overlap': allow_overlap}
        self.vocab = vocab
        self.suggester = suggester
        self.model = model
        self.name = name
        self.scorer = scorer
        self.add_negative_label = add_negative_label
        if not allow_overlap and max_positive is not None and (max_positive > 1):
            raise ValueError(Errors.E1051.format(max_positive=max_positive))

    @property
    def key(self) -> str:
        """Key of the doc.spans dict to save the spans under. During
        initialization and training, the component will look for spans on the
        reference document under the same key.
        """
        return str(self.cfg['spans_key'])

    def _allow_extra_label(self) -> None:
        """Raise an error if the component can not add any more labels."""
        nO = None
        if self.model.has_dim('nO'):
            nO = self.model.get_dim('nO')
        elif self.model.has_ref('output_layer') and self.model.get_ref('output_layer').has_dim('nO'):
            nO = self.model.get_ref('output_layer').get_dim('nO')
        if nO is not None and nO == self._n_labels:
            if not self.is_resizable:
                raise ValueError(Errors.E922.format(name=self.name, nO=self.model.get_dim('nO')))

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe.

        label (str): The label to add.
        RETURNS (int): 0 if label is already present, otherwise 1.

        DOCS: https://spacy.io/api/spancategorizer#add_label
        """
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self._allow_extra_label()
        self.cfg['labels'].append(label)
        self.vocab.strings.add(label)
        return 1

    @property
    def labels(self) -> Tuple[str]:
        """RETURNS (Tuple[str]): The labels currently added to the component.

        DOCS: https://spacy.io/api/spancategorizer#labels
        """
        return tuple(self.cfg['labels'])

    @property
    def label_data(self) -> List[str]:
        """RETURNS (List[str]): Information about the component's labels.

        DOCS: https://spacy.io/api/spancategorizer#label_data
        """
        return list(self.labels)

    @property
    def _label_map(self) -> Dict[str, int]:
        """RETURNS (Dict[str, int]): The label map."""
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def _n_labels(self) -> int:
        """RETURNS (int): Number of labels."""
        if self.add_negative_label:
            return len(self.labels) + 1
        else:
            return len(self.labels)

    @property
    def _negative_label_i(self) -> Union[int, None]:
        """RETURNS (Union[int, None]): Index of the negative label."""
        if self.add_negative_label:
            return len(self.label_data)
        else:
            return None

    def predict(self, docs: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The models prediction for each document.

        DOCS: https://spacy.io/api/spancategorizer#predict
        """
        indices = self.suggester(docs, ops=self.model.ops)
        if indices.lengths.sum() == 0:
            scores = self.model.ops.alloc2f(0, 0)
        else:
            scores = self.model.predict((docs, indices))
        return (indices, scores)

    def set_candidates(self, docs: Iterable[Doc], *, candidates_key: str='candidates') -> None:
        """Use the spancat suggester to add a list of span candidates to a list of docs.
        This method is intended to be used for debugging purposes.

        docs (Iterable[Doc]): The documents to modify.
        candidates_key (str): Key of the Doc.spans dict to save the candidate spans under.

        DOCS: https://spacy.io/api/spancategorizer#set_candidates
        """
        suggester_output = self.suggester(docs, ops=self.model.ops)
        for candidates, doc in zip(suggester_output, docs):
            doc.spans[candidates_key] = []
            for index in candidates.dataXd:
                doc.spans[candidates_key].append(doc[index[0]:index[1]])

    def set_annotations(self, docs: Iterable[Doc], indices_scores) -> None:
        """Modify a batch of Doc objects, using pre-computed scores.

        docs (Iterable[Doc]): The documents to modify.
        scores: The scores to set, produced by SpanCategorizer.predict.

        DOCS: https://spacy.io/api/spancategorizer#set_annotations
        """
        indices, scores = indices_scores
        offset = 0
        for i, doc in enumerate(docs):
            indices_i = indices[i].dataXd
            allow_overlap = cast(bool, self.cfg['allow_overlap'])
            if self.cfg['max_positive'] == 1:
                doc.spans[self.key] = self._make_span_group_singlelabel(doc, indices_i, scores[offset:offset + indices.lengths[i]], allow_overlap)
            else:
                doc.spans[self.key] = self._make_span_group_multilabel(doc, indices_i, scores[offset:offset + indices.lengths[i]])
            offset += indices.lengths[i]

    def update(self, examples: Iterable[Example], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss.

        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.

        DOCS: https://spacy.io/api/spancategorizer#update
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, 'SpanCategorizer.update')
        self._validate_categories(examples)
        if not any((len(eg.predicted) if eg.predicted else 0 for eg in examples)):
            return losses
        docs = [eg.predicted for eg in examples]
        spans = self.suggester(docs, ops=self.model.ops)
        if spans.lengths.sum() == 0:
            return losses
        set_dropout_rate(self.model, drop)
        scores, backprop_scores = self.model.begin_update((docs, spans))
        loss, d_scores = self.get_loss(examples, (spans, scores))
        backprop_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def get_loss(self, examples: Iterable[Example], spans_scores: Tuple[Ragged, Floats2d]) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.

        examples (Iterable[Examples]): The batch of examples.
        spans_scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.

        DOCS: https://spacy.io/api/spancategorizer#get_loss
        """
        spans, scores = spans_scores
        spans = Ragged(self.model.ops.to_numpy(spans.data), self.model.ops.to_numpy(spans.lengths))
        target = numpy.zeros(scores.shape, dtype=scores.dtype)
        if self.add_negative_label:
            negative_spans = numpy.ones(scores.shape[0])
        offset = 0
        label_map = self._label_map
        for i, eg in enumerate(examples):
            spans_index = {}
            spans_i = spans[i].dataXd
            for j in range(spans.lengths[i]):
                start = int(spans_i[j, 0])
                end = int(spans_i[j, 1])
                spans_index[start, end] = offset + j
            for gold_span in self._get_aligned_spans(eg):
                key = (gold_span.start, gold_span.end)
                if key in spans_index:
                    row = spans_index[key]
                    k = label_map[gold_span.label_]
                    target[row, k] = 1.0
                    if self.add_negative_label:
                        negative_spans[row] = 0.0
            offset += spans.lengths[i]
        target = self.model.ops.asarray(target, dtype='f')
        if self.add_negative_label:
            negative_samples = numpy.nonzero(negative_spans)[0]
            target[negative_samples, self._negative_label_i] = 1.0
        d_scores = scores - target
        if self.add_negative_label:
            neg_weight = cast(float, self.cfg['negative_weight'])
            if neg_weight != 1.0:
                d_scores[negative_samples] *= neg_weight
        loss = float((d_scores ** 2).sum())
        return (loss, d_scores)

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Optional[Language]=None, labels: Optional[List[str]]=None) -> None:
        """Initialize the pipe for training, using a representative set
        of data examples.

        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Optional[Language]): The current nlp object the component is part of.
        labels (Optional[List[str]]): The labels to add to the component, typically generated by the
            `init labels` command. If no labels are provided, the get_examples
            callback is used to extract the labels from the data.

        DOCS: https://spacy.io/api/spancategorizer#initialize
        """
        subbatch: List[Example] = []
        if labels is not None:
            for label in labels:
                self.add_label(label)
        for eg in get_examples():
            if labels is None:
                for span in eg.reference.spans.get(self.key, []):
                    self.add_label(span.label_)
            if len(subbatch) < 10:
                subbatch.append(eg)
        self._require_labels()
        if subbatch:
            docs = [eg.x for eg in subbatch]
            spans = build_ngram_suggester(sizes=[1])(docs)
            Y = self.model.ops.alloc2f(spans.dataXd.shape[0], self._n_labels)
            self.model.initialize(X=(docs, spans), Y=Y)
        else:
            self.model.initialize()

    def _validate_categories(self, examples: Iterable[Example]):
        pass

    def _get_aligned_spans(self, eg: Example):
        return eg.get_aligned_spans_y2x(eg.reference.spans.get(self.key, []), allow_overlap=True)

    def _make_span_group_multilabel(self, doc: Doc, indices: Ints2d, scores: Floats2d) -> SpanGroup:
        """Find the top-k labels for each span (k=max_positive)."""
        spans = SpanGroup(doc, name=self.key)
        if scores.size == 0:
            return spans
        scores = self.model.ops.to_numpy(scores)
        indices = self.model.ops.to_numpy(indices)
        threshold = self.cfg['threshold']
        max_positive = self.cfg['max_positive']
        keeps = scores >= threshold
        if max_positive is not None:
            assert isinstance(max_positive, int)
            if self.add_negative_label:
                negative_scores = numpy.copy(scores[:, self._negative_label_i])
                scores[:, self._negative_label_i] = -numpy.inf
                ranked = (scores * -1).argsort()
                scores[:, self._negative_label_i] = negative_scores
            else:
                ranked = (scores * -1).argsort()
            span_filter = ranked[:, max_positive:]
            for i, row in enumerate(span_filter):
                keeps[i, row] = False
        attrs_scores = []
        for i in range(indices.shape[0]):
            start = indices[i, 0]
            end = indices[i, 1]
            for j, keep in enumerate(keeps[i]):
                if keep:
                    if j != self._negative_label_i:
                        spans.append(Span(doc, start, end, label=self.labels[j]))
                        attrs_scores.append(scores[i, j])
        spans.attrs['scores'] = numpy.array(attrs_scores)
        return spans

    def _make_span_group_singlelabel(self, doc: Doc, indices: Ints2d, scores: Floats2d, allow_overlap: bool=True) -> SpanGroup:
        """Find the argmax label for each span."""
        if scores.size == 0:
            return SpanGroup(doc, name=self.key)
        scores = self.model.ops.to_numpy(scores)
        indices = self.model.ops.to_numpy(indices)
        predicted = scores.argmax(axis=1)
        argmax_scores = numpy.take_along_axis(scores, numpy.expand_dims(predicted, 1), axis=1)
        keeps = numpy.ones(predicted.shape, dtype=bool)
        if self.add_negative_label:
            keeps = numpy.logical_and(keeps, predicted != self._negative_label_i)
        threshold = self.cfg['threshold']
        if threshold is not None:
            keeps = numpy.logical_and(keeps, (argmax_scores >= threshold).squeeze())
        if not allow_overlap:
            sort_idx = (argmax_scores.squeeze() * -1).argsort()
            argmax_scores = argmax_scores[sort_idx]
            predicted = predicted[sort_idx]
            indices = indices[sort_idx]
            keeps = keeps[sort_idx]
        seen = _Intervals()
        spans = SpanGroup(doc, name=self.key)
        attrs_scores = []
        for i in range(indices.shape[0]):
            if not keeps[i]:
                continue
            label = predicted[i]
            start = indices[i, 0]
            end = indices[i, 1]
            if not allow_overlap:
                if (start, end) in seen:
                    continue
                else:
                    seen.add(start, end)
            attrs_scores.append(argmax_scores[i])
            spans.append(Span(doc, start, end, label=self.labels[label]))
        spans.attrs['scores'] = numpy.array(attrs_scores)
        return spans