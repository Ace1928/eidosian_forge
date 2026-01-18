from typing import Callable, Iterable, Iterator
import pytest
from thinc.api import Config
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from spacy.training.loop import train
from spacy.util import load_model_from_config, registry
class UnannotatedCorpus:

    def __call__(self, nlp: Language) -> Iterator[Example]:
        for text in ['a a', 'b b', 'c c']:
            doc = nlp.make_doc(text)
            yield Example(doc, doc)