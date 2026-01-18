import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class _VarTableVariant(Union):
    """Must be in an anonymous union or values will not work across various VT's."""
    _fields_ = [('bstrVal', LPCWSTR)]