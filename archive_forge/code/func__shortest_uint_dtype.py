from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
def _shortest_uint_dtype(max_value):
    """Get the shortest unsingned integer data-type required for representing values up to a given
    maximum value.

    Returns the shortest unsingned integer data-type required for representing values up to a given
    maximum value.

    Parameters
    ----------
    max_value : int
        The maximum value we wish to represent.

    Returns
    -------
    data-type
        The shortest unsigned integer data-type required for representing values up to a given
        maximum value.
    """
    if max_value < 2 ** 8:
        return np.uint8
    elif max_value < 2 ** 16:
        return np.uint16
    elif max_value < 2 ** 32:
        return np.uint32
    return np.uint64