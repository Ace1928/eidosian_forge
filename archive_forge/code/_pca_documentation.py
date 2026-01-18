from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.special import gammaln
from ..base import _fit_context
from ..utils import check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._array_api import _convert_to_numpy, get_namespace
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.extmath import fast_logdet, randomized_svd, stable_cumsum, svd_flip
from ..utils.sparsefuncs import _implicit_column_offset, mean_variance_axis
from ..utils.validation import check_is_fitted
from ._base import _BasePCA
Return the average log-likelihood of all samples.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        y : Ignored
            Ignored.

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        