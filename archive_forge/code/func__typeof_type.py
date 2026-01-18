from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(type)
def _typeof_type(val, c):
    """
    Type various specific Python types.
    """
    if issubclass(val, BaseException):
        return types.ExceptionClass(val)
    if issubclass(val, tuple) and hasattr(val, '_asdict'):
        return types.NamedTupleClass(val)
    if issubclass(val, np.generic):
        return types.NumberClass(numpy_support.from_dtype(val))
    if issubclass(val, types.Type):
        return types.TypeRef(val)
    from numba.typed import Dict
    if issubclass(val, Dict):
        return types.TypeRef(types.DictType)
    from numba.typed import List
    if issubclass(val, List):
        return types.TypeRef(types.ListType)