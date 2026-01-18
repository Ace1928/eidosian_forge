import operator
import sys
import time
import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import _fit_context
from ..exceptions import ConvergenceWarning
from ..linear_model import _cd_fast as cd_fast  # type: ignore
from ..linear_model import lars_path_gram
from ..model_selection import check_cv, cross_val_score
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import EmpiricalCovariance, empirical_covariance, log_likelihood
def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return gap