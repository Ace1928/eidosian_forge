from typing import Callable, Iterable, Iterator
import pytest
from thinc.api import Config
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from spacy.training.loop import train
from spacy.util import load_model_from_config, registry
class AssertSents:

    def __init__(self, name, **cfg):
        self.name = name
        pass

    def __call__(self, doc):
        if not doc.has_annotation('SENT_START'):
            raise ValueError('No sents')
        return doc

    def update(self, examples, *, drop=0.0, sgd=None, losses=None):
        for example in examples:
            if not example.predicted.has_annotation('SENT_START'):
                raise ValueError('No sents')
        return {}