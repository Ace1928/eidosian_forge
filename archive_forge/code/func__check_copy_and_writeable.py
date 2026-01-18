import sys
import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import interpolate, linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..model_selection import check_cv
from ..utils import (  # type: ignore
from ..utils._metadata_requests import (
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, LinearRegression, _preprocess_data
def _check_copy_and_writeable(array, copy=False):
    if copy or not array.flags.writeable:
        return array.copy()
    return array