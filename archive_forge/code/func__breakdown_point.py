import warnings
from itertools import combinations
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import binom
from ..base import RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel
def _breakdown_point(n_samples, n_subsamples):
    """Approximation of the breakdown point.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    n_subsamples : int
        Number of subsamples to consider.

    Returns
    -------
    breakdown_point : float
        Approximation of breakdown point.
    """
    return 1 - (0.5 ** (1 / n_subsamples) * (n_samples - n_subsamples + 1) + n_subsamples - 1) / n_samples