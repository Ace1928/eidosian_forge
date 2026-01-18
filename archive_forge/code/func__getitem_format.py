from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def _getitem_format(self, s_result):
    if self.sparse_serialization:
        if self.gensim:
            s_result = self._getitem_sparse2gensim(s_result)
        elif not self.sparse_retrieval:
            s_result = numpy.array(s_result.todense())
    elif self.gensim:
        s_result = self._getitem_dense2gensim(s_result)
    elif self.sparse_retrieval:
        s_result = sparse.csr_matrix(s_result)
    return s_result