import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
@dataclass
class Doctag:
    """A dataclass shape-compatible with keyedvectors.SimpleVocab, extended to record
    details of string document tags discovered during the initial vocabulary scan.

    Will not be used if all presented document tags are ints. No longer used in a
    completed model: just used during initial scan, and for backward compatibility.
    """
    __slots__ = ('doc_count', 'index', 'word_count')
    doc_count: int
    index: int
    word_count: int

    @property
    def count(self):
        return self.doc_count

    @count.setter
    def count(self, new_val):
        self.doc_count = new_val