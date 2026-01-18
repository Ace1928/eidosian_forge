import ctypes
from numbers import Number, Integral
from ...base import get_last_ffi_error, _LIB
from ..base import c_str
from .types import MXNetValue, TypeCode
from .types import RETURN_SWITCH
from .object import ObjectBase
from ..node_generic import convert_to_node
from ..._ctypes.ndarray import NDArrayBase
def _make_mxnet_args(args, temp_args):
    """Pack arguments into c args mxnet call accept"""
    num_args = len(args)
    values = (MXNetValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if isinstance(arg, ObjectBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
        elif arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, str):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, (list, tuple)):
            arg = convert_to_node(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, NDArrayBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NDARRAYHANDLE
        elif isinstance(arg, ctypes.c_void_p):
            values[i].v_handle = arg
            type_codes[i] = TypeCode.HANDLE
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
    return (values, type_codes, num_args)