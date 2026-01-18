import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def _symmetrize(self):
    """Word pairs may have been encountered in (i, j) and (j, i) order.

        Notes
        -----
        Rather than enforcing a particular ordering during the update process,
        we choose to symmetrize the co-occurrence matrix after accumulation has completed.

        """
    co_occ = self._co_occurrences
    co_occ.setdiag(self._occurrences)
    self._co_occurrences = co_occ + co_occ.T - sps.diags(co_occ.diagonal(), offsets=0, dtype='uint32')