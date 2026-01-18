import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _cpu_features(self):
    target = self.target_name
    if target == 'baseline':
        target = __cpu_baseline__
    else:
        target = target.split('__')
    return ' '.join(target)