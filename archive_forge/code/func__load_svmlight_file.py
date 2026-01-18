import os.path
from contextlib import closing
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .. import __version__
from ..utils import IS_PYPY, check_array
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
def _load_svmlight_file(*args, **kwargs):
    raise NotImplementedError('load_svmlight_file is currently not compatible with PyPy (see https://github.com/scikit-learn/scikit-learn/issues/11543 for the status updates).')