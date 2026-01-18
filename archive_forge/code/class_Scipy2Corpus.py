from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
class Scipy2Corpus:
    """Convert a sequence of dense/sparse vectors into a streamed Gensim corpus object.

    See Also
    --------
    :func:`~gensim.matutils.corpus2csc`
        Convert corpus in Gensim format to `scipy.sparse.csc` matrix.

    """

    def __init__(self, vecs):
        """

        Parameters
        ----------
        vecs : iterable of {`numpy.ndarray`, `scipy.sparse`}
            Input vectors.

        """
        self.vecs = vecs

    def __iter__(self):
        for vec in self.vecs:
            if isinstance(vec, np.ndarray):
                yield full2sparse(vec)
            else:
                yield scipy2sparse(vec)

    def __len__(self):
        return len(self.vecs)