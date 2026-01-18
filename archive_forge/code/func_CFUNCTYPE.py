import os as _os, sys as _sys
import types as _types
from _ctypes import Union, Structure, Array
from _ctypes import _Pointer
from _ctypes import CFuncPtr as _CFuncPtr
from _ctypes import __version__ as _ctypes_version
from _ctypes import RTLD_LOCAL, RTLD_GLOBAL
from _ctypes import ArgumentError
from struct import calcsize as _calcsize
from _ctypes import FUNCFLAG_CDECL as _FUNCFLAG_CDECL, \
from _ctypes import sizeof, byref, addressof, alignment, resize
from _ctypes import get_errno, set_errno
from _ctypes import _SimpleCData
from _ctypes import POINTER, pointer, _pointer_type_cache
from _ctypes import _memmove_addr, _memset_addr, _string_at_addr, _cast_addr
from ctypes._endian import BigEndianStructure, LittleEndianStructure
from ctypes._endian import BigEndianUnion, LittleEndianUnion
def CFUNCTYPE(restype, *argtypes, **kw):
    """CFUNCTYPE(restype, *argtypes,
                 use_errno=False, use_last_error=False) -> function prototype.

    restype: the result type
    argtypes: a sequence specifying the argument types

    The function prototype can be called in different ways to create a
    callable object:

    prototype(integer address) -> foreign function
    prototype(callable) -> create and return a C callable function from callable
    prototype(integer index, method name[, paramflags]) -> foreign function calling a COM method
    prototype((ordinal number, dll object)[, paramflags]) -> foreign function exported by ordinal
    prototype((function name, dll object)[, paramflags]) -> foreign function exported by name
    """
    flags = _FUNCFLAG_CDECL
    if kw.pop('use_errno', False):
        flags |= _FUNCFLAG_USE_ERRNO
    if kw.pop('use_last_error', False):
        flags |= _FUNCFLAG_USE_LASTERROR
    if kw:
        raise ValueError('unexpected keyword argument(s) %s' % kw.keys())
    try:
        return _c_functype_cache[restype, argtypes, flags]
    except KeyError:
        pass

    class CFunctionType(_CFuncPtr):
        _argtypes_ = argtypes
        _restype_ = restype
        _flags_ = flags
    _c_functype_cache[restype, argtypes, flags] = CFunctionType
    return CFunctionType