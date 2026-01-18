import os.path
from contextlib import closing
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .. import __version__
from ..utils import IS_PYPY, check_array
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
def _open_and_load(f, dtype, multilabel, zero_based, query_id, offset=0, length=-1):
    if hasattr(f, 'read'):
        actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(f, dtype, multilabel, zero_based, query_id, offset, length)
    else:
        with closing(_gen_open(f)) as f:
            actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(f, dtype, multilabel, zero_based, query_id, offset, length)
    if not multilabel:
        labels = np.frombuffer(labels, np.float64)
    data = np.frombuffer(data, actual_dtype)
    indices = np.frombuffer(ind, np.longlong)
    indptr = np.frombuffer(indptr, dtype=np.longlong)
    query = np.frombuffer(query, np.int64)
    data = np.asarray(data, dtype=dtype)
    return (data, indices, indptr, labels, query)