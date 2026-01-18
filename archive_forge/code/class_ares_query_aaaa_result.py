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
class ares_query_aaaa_result(AresResult):
    __slots__ = ('host', 'ttl')
    type = 'AAAA'

    def __init__(self, ares_addrttl):
        buf = _ffi.new('char[]', _lib.INET6_ADDRSTRLEN)
        _lib.ares_inet_ntop(socket.AF_INET6, _ffi.addressof(ares_addrttl.ip6addr), buf, _lib.INET6_ADDRSTRLEN)
        self.host = maybe_str(_ffi.string(buf, _lib.INET6_ADDRSTRLEN))
        self.ttl = ares_addrttl.ttl