import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _int_max(self):
    if self._is_fp():
        return None
    max_u = self._to_unsigned(self.setall(-1))[0]
    if self._is_signed():
        return max_u // 2
    return max_u