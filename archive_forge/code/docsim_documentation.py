import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly; use the `self[query]` syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix (if maintain_sparsity=False) **OR**
        :class:`scipy.sparse.csc`
            otherwise

        