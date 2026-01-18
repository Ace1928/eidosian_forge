import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _to_unsigned(self, vector):
    if isinstance(vector, (list, tuple)):
        return getattr(self.npyv, 'load_u' + self.sfx[1:])(vector)
    else:
        sfx = vector.__name__.replace('npyv_', '')
        if sfx[0] == 'b':
            cvt_intrin = 'cvt_u{0}_b{0}'
        else:
            cvt_intrin = 'reinterpret_u{0}_{1}'
        return getattr(self.npyv, cvt_intrin.format(sfx[1:], sfx))(vector)