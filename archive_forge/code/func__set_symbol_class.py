import ctypes
from ..base import _LIB
from ..base import c_str_array, c_handle_array, c_str, mx_uint
from ..base import SymbolHandle
from ..base import check_call
def _set_symbol_class(cls):
    """Set the symbolic class to be cls"""
    global _symbol_cls
    _symbol_cls = cls