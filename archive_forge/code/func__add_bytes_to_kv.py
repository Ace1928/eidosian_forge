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
def _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding):
    start = 0
    processed_words = 0
    bytes_per_vector = vector_size * dtype(REAL).itemsize
    max_words = vocab_size - kv.next_index
    assert max_words > 0
    for _ in range(max_words):
        i_space = chunk.find(b' ', start)
        i_vector = i_space + 1
        if i_space == -1 or len(chunk) - i_vector < bytes_per_vector:
            break
        word = chunk[start:i_space].decode(encoding, errors=unicode_errors)
        word = word.lstrip('\n')
        vector = frombuffer(chunk, offset=i_vector, count=vector_size, dtype=REAL).astype(datatype)
        _add_word_to_kv(kv, counts, word, vector, vocab_size)
        start = i_vector + bytes_per_vector
        processed_words += 1
    return (processed_words, chunk[start:])