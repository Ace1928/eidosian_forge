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
class ares_query_ptr_result(AresResult):
    __slots__ = ('name', 'ttl', 'aliases')
    type = 'PTR'

    def __init__(self, hostent, aliases):
        self.name = maybe_str(_ffi.string(hostent.h_name))
        self.aliases = aliases
        self.ttl = -1