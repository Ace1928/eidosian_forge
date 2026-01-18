from __future__ import annotations
from types import TracebackType
from typing import Any, ClassVar, cast
class CTraceback(ctypes.Structure):
    _fields_: ClassVar = [('PyObject_HEAD', ctypes.c_byte * object().__sizeof__()), ('tb_next', ctypes.c_void_p), ('tb_frame', ctypes.c_void_p), ('tb_lasti', ctypes.c_int), ('tb_lineno', ctypes.c_int)]