from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
def _normalize_dense_vector(vector, matrix, normalization):
    """Normalize a dense vector after a change of basis.

    Parameters
    ----------
    vector : 1xN ndarray
        A dense vector.
    matrix : NxN ndarray
        A change-of-basis matrix.
    normalization : {True, False, 'maintain'}
        Whether the vector will be L2-normalized (True; corresponds to the soft
        cosine measure), maintain its L2-norm during the change of basis
        ('maintain'; corresponds to query expansion with partial membership),
        or kept as-is (False; corresponds to query expansion).

    Returns
    -------
    vector : ndarray
        The normalized dense vector.

    """
    if not normalization:
        return vector
    vector_norm = vector.T.dot(matrix).dot(vector)[0, 0]
    assert vector_norm >= 0.0, NON_NEGATIVE_NORM_ASSERTION_MESSAGE
    if normalization == 'maintain' and vector_norm > 0.0:
        vector_norm /= vector.T.dot(vector)
    vector_norm = sqrt(vector_norm)
    normalized_vector = vector
    if vector_norm > 0.0:
        normalized_vector /= vector_norm
    return normalized_vector