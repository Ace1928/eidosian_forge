from numbers import Integral, Real
import numpy as np
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from ..base import (
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import stable_cumsum
from ..utils.validation import FLOAT_DTYPES, check_is_fitted

        Transform new points into embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Returns the instance itself.

        Notes
        -----
        Because of scaling performed by this method, it is discouraged to use
        it together with methods that are not scale-invariant (like SVMs).
        