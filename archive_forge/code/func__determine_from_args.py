import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _determine_from_args(self, args, casting):
    n = len(args)
    in_types = tuple((arg.dtype for arg in args))
    for op in self._ops:
        op_types = op.in_types
        for i in range(n):
            it = in_types[i]
            ot = op_types[i]
            if not numpy.can_cast(it, ot, casting=casting):
                break
        else:
            return op
    return None