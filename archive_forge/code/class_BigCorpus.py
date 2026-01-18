import logging
import unittest
import os
import numpy as np
import gensim
from gensim.test.utils import get_tmpfile
class BigCorpus:
    """A corpus of a large number of docs & large vocab"""

    def __init__(self, words_only=False, num_terms=200000, num_docs=1000000, doc_len=100):
        self.dictionary = gensim.utils.FakeDict(num_terms)
        self.words_only = words_only
        self.num_docs = num_docs
        self.doc_len = doc_len

    def __iter__(self):
        for _ in range(self.num_docs):
            doc_len = np.random.poisson(self.doc_len)
            ids = np.random.randint(0, len(self.dictionary), doc_len)
            if self.words_only:
                yield [str(idx) for idx in ids]
            else:
                weights = np.random.poisson(3, doc_len)
                yield sorted(zip(ids, weights))