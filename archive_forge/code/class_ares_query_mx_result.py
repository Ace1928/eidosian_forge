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
class ares_query_mx_result(AresResult):
    __slots__ = ('host', 'priority', 'ttl')
    type = 'MX'

    def __init__(self, mx):
        self.host = maybe_str(_ffi.string(mx.host))
        self.priority = mx.priority
        self.ttl = -1