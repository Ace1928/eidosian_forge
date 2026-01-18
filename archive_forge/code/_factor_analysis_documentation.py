import warnings
from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import fast_logdet, randomized_svd, squared_norm
from ..utils.validation import check_is_fitted
Number of transformed output features.