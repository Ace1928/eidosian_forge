from typing import Dict
from ml_dtypes._custom_floats import bfloat16
from ml_dtypes._custom_floats import float8_e4m3b11fnuz
from ml_dtypes._custom_floats import float8_e4m3fn
from ml_dtypes._custom_floats import float8_e4m3fnuz
from ml_dtypes._custom_floats import float8_e5m2
from ml_dtypes._custom_floats import float8_e5m2fnuz
import numpy as np
@staticmethod
def _float8_e4m3fn_finfo():

    def float_to_str(f):
        return '%6.2e' % float(f)
    tiny = float.fromhex('0x1p-6')
    resolution = 0.1
    eps = float.fromhex('0x1p-3')
    epsneg = float.fromhex('0x1p-4')
    max_ = float.fromhex('0x1.Cp8')
    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3fn_dtype
    obj.bits = 8
    obj.eps = float8_e4m3fn(eps)
    obj.epsneg = float8_e4m3fn(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3fn(max_)
    obj.min = float8_e4m3fn(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 9
    obj.minexp = -6
    obj.precision = 1
    obj.resolution = float8_e4m3fn(resolution)
    obj._machar = _Float8E4m3fnMachArLike()
    if not hasattr(obj, 'tiny'):
        obj.tiny = float8_e4m3fn(tiny)
    if not hasattr(obj, 'smallest_normal'):
        obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal
    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    return obj