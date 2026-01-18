from warnings import warn
import numpy as np
from numpy import asarray
from scipy.sparse import (issparse,
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.linalg import LinAlgError
import copy
from . import _superlu
def _get_umf_family(A):
    """Get umfpack family string given the sparse matrix dtype."""
    _families = {(np.float64, np.int32): 'di', (np.complex128, np.int32): 'zi', (np.float64, np.int64): 'dl', (np.complex128, np.int64): 'zl'}
    f_type = getattr(np, A.dtype.name)
    i_type = getattr(np, A.indices.dtype.name)
    try:
        family = _families[f_type, i_type]
    except KeyError as e:
        msg = f'only float64 or complex128 matrices with int32 or int64 indices are supported! (got: matrix: {f_type}, indices: {i_type})'
        raise ValueError(msg) from e
    family = family[0] + 'l'
    A_new = copy.copy(A)
    A_new.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    A_new.indices = np.array(A.indices, copy=False, dtype=np.int64)
    return (family, A_new)