import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _determine_from_dtype(self, dtype):
    for op in self._ops:
        op_types = op.out_types
        for t in op_types:
            if t != dtype:
                break
        else:
            return op
    return None