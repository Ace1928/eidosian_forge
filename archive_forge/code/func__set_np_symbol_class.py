import ctypes
from ..base import _LIB
from ..base import c_str_array, c_handle_array, c_str, mx_uint
from ..base import SymbolHandle
from ..base import check_call
def _set_np_symbol_class(cls):
    """Set the numpy-compatible symbolic class to be cls"""
    global _np_symbol_cls
    _np_symbol_cls = cls