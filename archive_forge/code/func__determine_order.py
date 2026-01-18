import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _determine_order(self, args, order):
    if order.upper() in ('C', 'K'):
        return 'C'
    elif order.upper() == 'A':
        order = 'F' if all([a.flags.f_contiguous and (not a.flags.c_contiguous) for a in args]) else 'C'
        return order
    elif order.upper() == 'F':
        return 'F'
    else:
        raise RuntimeError(f'Unknown order {order}')