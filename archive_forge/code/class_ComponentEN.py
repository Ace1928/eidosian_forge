import pytest
from thinc.api import ConfigValidationError, Linear, Model
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, combine_score_weights, registry
from ..util import make_tempdir
@English.factory(name)
class ComponentEN:

    def __init__(self, nlp: Language, name: str, value1: StrictInt, value2: StrictStr):
        self.nlp = nlp
        self.value1 = value1
        self.value2 = value2
        self.is_base = False

    def __call__(self, doc: Doc) -> Doc:
        return doc