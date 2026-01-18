import scipy.linalg._interpolative_backend as _backend
import numpy as np
import sys
def _is_real(A):
    try:
        if A.dtype == np.complex128:
            return False
        elif A.dtype == np.float64:
            return True
        else:
            raise _DTYPE_ERROR
    except AttributeError as e:
        raise _TYPE_ERROR from e