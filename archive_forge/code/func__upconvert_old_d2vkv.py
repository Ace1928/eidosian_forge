import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def _upconvert_old_d2vkv(self):
    """Convert a deserialized older Doc2VecKeyedVectors instance to latest generic KeyedVectors"""
    self.vocab = self.doctags
    self._upconvert_old_vocab()
    for k in self.key_to_index.keys():
        old_offset = self.get_vecattr(k, 'offset')
        true_index = old_offset + self.max_rawint + 1
        self.key_to_index[k] = true_index
    del self.expandos['offset']
    if self.max_rawint > -1:
        self.index_to_key = list(range(0, self.max_rawint + 1)) + self.offset2doctag
    else:
        self.index_to_key = self.offset2doctag
    self.vectors = self.vectors_docs
    del self.doctags
    del self.vectors_docs
    del self.count
    del self.max_rawint
    del self.offset2doctag