import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
Return a marginal KDE distribution

        Parameters
        ----------
        dimensions : int or 1-d array_like
            The dimensions of the multivariate distribution corresponding
            with the marginal variables, that is, the indices of the dimensions
            that are being retained. The other dimensions are marginalized out.

        Returns
        -------
        marginal_kde : gaussian_kde
            An object representing the marginal distribution.

        Notes
        -----
        .. versionadded:: 1.10.0

        