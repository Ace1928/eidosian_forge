from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
def _normalize_dense_corpus(corpus, matrix, normalization):
    """Normalize a dense corpus after a change of basis.

    Parameters
    ----------
    corpus : MxN ndarray
        A dense corpus.
    matrix : NxN ndarray
        A change-of-basis matrix.
    normalization : {True, False, 'maintain'}
        Whether the vector will be L2-normalized (True; corresponds to the soft
        cosine measure), maintain its L2-norm during the change of basis
        ('maintain'; corresponds to query expansion with partial membership),
        or kept as-is (False; corresponds to query expansion).

    Returns
    -------
    normalized_corpus : ndarray
        The normalized dense corpus.

    """
    if not normalization:
        return corpus
    corpus_norm = np.multiply(corpus.T.dot(matrix), corpus.T).sum(axis=1).T
    assert corpus_norm.min() >= 0.0, NON_NEGATIVE_NORM_ASSERTION_MESSAGE
    if normalization == 'maintain':
        corpus_norm /= np.multiply(corpus.T, corpus.T).sum(axis=1).T
    corpus_norm = np.sqrt(corpus_norm)
    normalized_corpus = np.multiply(corpus, 1.0 / corpus_norm)
    normalized_corpus = np.nan_to_num(normalized_corpus)
    return normalized_corpus