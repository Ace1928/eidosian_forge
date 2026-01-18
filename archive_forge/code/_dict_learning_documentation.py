import itertools
import sys
import time
from numbers import Integral, Real
from warnings import warn
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from ..base import (
from ..linear_model import Lars, Lasso, LassoLars, orthogonal_mp_gram
from ..utils import check_array, check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.extmath import randomized_svd, row_norms, svd_flip
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
Number of transformed output features.