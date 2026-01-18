import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _arg_wlen_as_expected(value):
    """Ensure argument `wlen` is of type `np.intp` and larger than 1.

    Used in `peak_prominences` and `peak_widths`.

    Returns
    -------
    value : np.intp
        The original `value` rounded up to an integer or -1 if `value` was
        None.
    """
    if value is None:
        value = -1
    elif 1 < value:
        if not cupy.can_cast(value, cupy.int64, 'safe'):
            value = math.ceil(value)
        value = int(value)
    else:
        raise ValueError('`wlen` must be larger than 1, was {}'.format(value))
    return value