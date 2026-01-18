import ctypes
from numbers import Number, Integral
from ...base import get_last_ffi_error, _LIB
from ..base import c_str
from .types import MXNetValue, TypeCode
from .types import RETURN_SWITCH
from .object import ObjectBase
from ..node_generic import convert_to_node
from ..._ctypes.ndarray import NDArrayBase
def _set_class_object(obj_class):
    global _CLASS_OBJECT
    _CLASS_OBJECT = obj_class