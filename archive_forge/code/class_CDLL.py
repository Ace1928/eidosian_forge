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
class CDLL(object):
    """An instance of this class represents a loaded dll/shared
    library, exporting functions using the standard C calling
    convention (named 'cdecl' on Windows).

    The exported functions can be accessed as attributes, or by
    indexing with the function name.  Examples:

    <obj>.qsort -> callable object
    <obj>['qsort'] -> callable object

    Calling the functions releases the Python GIL during the call and
    reacquires it afterwards.
    """
    _func_flags_ = _FUNCFLAG_CDECL
    _func_restype_ = c_int
    _name = '<uninitialized>'
    _handle = 0
    _FuncPtr = None

    def __init__(self, name, mode=DEFAULT_MODE, handle=None, use_errno=False, use_last_error=False, winmode=None):
        self._name = name
        flags = self._func_flags_
        if use_errno:
            flags |= _FUNCFLAG_USE_ERRNO
        if use_last_error:
            flags |= _FUNCFLAG_USE_LASTERROR
        if _sys.platform.startswith('aix'):
            'When the name contains ".a(" and ends with ")",\n               e.g., "libFOO.a(libFOO.so)" - this is taken to be an\n               archive(member) syntax for dlopen(), and the mode is adjusted.\n               Otherwise, name is presented to dlopen() as a file argument.\n            '
            if name and name.endswith(')') and ('.a(' in name):
                mode |= _os.RTLD_MEMBER | _os.RTLD_NOW
        if _os.name == 'nt':
            if winmode is not None:
                mode = winmode
            else:
                import nt
                mode = nt._LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                if '/' in name or '\\' in name:
                    self._name = nt._getfullpathname(self._name)
                    mode |= nt._LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR

        class _FuncPtr(_CFuncPtr):
            _flags_ = flags
            _restype_ = self._func_restype_
        self._FuncPtr = _FuncPtr
        if handle is None:
            self._handle = _dlopen(self._name, mode)
        else:
            self._handle = handle

    def __repr__(self):
        return "<%s '%s', handle %x at %#x>" % (self.__class__.__name__, self._name, self._handle & _sys.maxsize * 2 + 1, id(self) & _sys.maxsize * 2 + 1)

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        func = self.__getitem__(name)
        setattr(self, name, func)
        return func

    def __getitem__(self, name_or_ordinal):
        func = self._FuncPtr((name_or_ordinal, self))
        if not isinstance(name_or_ordinal, int):
            func.__name__ = name_or_ordinal
        return func