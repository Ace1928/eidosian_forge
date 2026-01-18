import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _x2(self, intrin_name):
    return getattr(self.npyv, f'{intrin_name}_{self.sfx}x2')