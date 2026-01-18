import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _isdtype_single(dtype, kind, *, xp):
    if isinstance(kind, str):
        if kind == 'bool':
            return dtype == xp.bool
        elif kind == 'signed integer':
            return dtype in {xp.int8, xp.int16, xp.int32, xp.int64}
        elif kind == 'unsigned integer':
            return dtype in {xp.uint8, xp.uint16, xp.uint32, xp.uint64}
        elif kind == 'integral':
            return any((_isdtype_single(dtype, k, xp=xp) for k in ('signed integer', 'unsigned integer')))
        elif kind == 'real floating':
            return dtype in supported_float_dtypes(xp)
        elif kind == 'complex floating':
            complex_dtypes = set()
            if hasattr(xp, 'complex64'):
                complex_dtypes.add(xp.complex64)
            if hasattr(xp, 'complex128'):
                complex_dtypes.add(xp.complex128)
            return dtype in complex_dtypes
        elif kind == 'numeric':
            return any((_isdtype_single(dtype, k, xp=xp) for k in ('integral', 'real floating', 'complex floating')))
        else:
            raise ValueError(f'Unrecognized data type kind: {kind!r}')
    else:
        return dtype == kind