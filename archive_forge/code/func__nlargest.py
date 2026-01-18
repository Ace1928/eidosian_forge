import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def _nlargest(n, iterable):
    """Helper for extracting n documents with maximum similarity.

    Parameters
    ----------
    n : int
        Number of elements to be extracted
    iterable : iterable of list of (int, float)
        Iterable containing documents with computed similarities

    Returns
    -------
    :class:`list`
        List with the n largest elements from the dataset defined by iterable.

    Notes
    -----
    Elements are compared by the absolute value of similarity, because negative value of similarity
    does not mean some form of dissimilarity.

    """
    return heapq.nlargest(n, itertools.chain(*iterable), key=lambda item: abs(item[1]))