from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def _to_float_str(val, exp=0):
    if isinstance(val, float):
        if math.isnan(val):
            res = 'NaN'
        elif val == 0.0:
            sone = math.copysign(1.0, val)
            if sone < 0.0:
                return '-0.0'
            else:
                return '+0.0'
        elif val == float('+inf'):
            res = '+oo'
        elif val == float('-inf'):
            res = '-oo'
        else:
            v = val.as_integer_ratio()
            num = v[0]
            den = v[1]
            rvs = str(num) + '/' + str(den)
            res = rvs + 'p' + _to_int_str(exp)
    elif isinstance(val, bool):
        if val:
            res = '1.0'
        else:
            res = '0.0'
    elif _is_int(val):
        res = str(val)
    elif isinstance(val, str):
        inx = val.find('*(2**')
        if inx == -1:
            res = val
        elif val[-1] == ')':
            res = val[0:inx]
            exp = str(int(val[inx + 5:-1]) + int(exp))
        else:
            _z3_assert(False, 'String does not have floating-point numeral form.')
    elif z3_debug():
        _z3_assert(False, 'Python value cannot be used to create floating-point numerals.')
    if exp == 0:
        return res
    else:
        return res + 'p' + exp