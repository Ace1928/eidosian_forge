import warnings
from unittest import TestCase
import pytest
import srsly
from numpy import zeros
from spacy.kb.kb_in_memory import InMemoryLookupKB, Writer
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
class MyPipe(TrainablePipe):

    def __init__(self, vocab, model=True, **cfg):
        if cfg:
            self.cfg = cfg
        else:
            self.cfg = None
        self.model = SerializableDummy()
        self.vocab = vocab