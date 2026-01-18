import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _arg_peaks_as_expected(value):
    """Ensure argument `peaks` is a 1-D C-contiguous array of dtype('int64').

    Used in `peak_prominences` and `peak_widths` to make `peaks` compatible
    with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('int64').
    """
    value = cupy.asarray(value)
    if value.size == 0:
        value = cupy.array([], dtype=cupy.int64)
    try:
        value = value.astype(cupy.int64, order='C', copy=False)
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    if value.ndim != 1:
        raise ValueError('`peaks` must be a 1-D array')
    return value