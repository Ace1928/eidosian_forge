from ._cares import ffi as _ffi, lib as _lib
import _cffi_backend  # hint for bundler tools
from . import errno
from .utils import ascii_bytes, maybe_str, parse_name
from ._version import __version__
import collections.abc
import socket
import math
import functools
import sys
class AresResult:
    __slots__ = ()

    def __repr__(self):
        attrs = ['%s=%s' % (a, getattr(self, a)) for a in self.__slots__]
        return '<%s> %s' % (self.__class__.__name__, ', '.join(attrs))