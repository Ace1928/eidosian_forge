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
def _getitem_dense2gensim(self, result):
    """Change given dense result matrix to gensim sparse vectors."""
    if len(result.shape) == 1:
        output = gensim.matutils.full2sparse(result)
    else:
        output = (gensim.matutils.full2sparse(result[i]) for i in range(result.shape[0]))
    return output