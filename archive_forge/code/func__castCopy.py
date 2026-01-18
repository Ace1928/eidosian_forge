import numpy
from numpy import asarray_chkfinite, single, asarray, array
from numpy.linalg import norm
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from ._decomp import eigvals
def _castCopy(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.char == type:
            cast_arrays = cast_arrays + (a.copy(),)
        else:
            cast_arrays = cast_arrays + (a.astype(type),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays