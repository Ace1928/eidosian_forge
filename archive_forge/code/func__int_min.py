import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _int_min(self):
    if self._is_fp():
        return None
    if self._is_unsigned():
        return 0
    return -(self._int_max() + 1)