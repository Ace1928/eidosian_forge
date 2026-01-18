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
class ares_host_result(AresResult):
    __slots__ = ('name', 'aliases', 'addresses')

    def __init__(self, hostent):
        self.name = maybe_str(_ffi.string(hostent.h_name))
        self.aliases = []
        self.addresses = []
        i = 0
        while hostent.h_aliases[i] != _ffi.NULL:
            self.aliases.append(maybe_str(_ffi.string(hostent.h_aliases[i])))
            i += 1
        i = 0
        while hostent.h_addr_list[i] != _ffi.NULL:
            buf = _ffi.new('char[]', _lib.INET6_ADDRSTRLEN)
            if _ffi.NULL != _lib.ares_inet_ntop(hostent.h_addrtype, hostent.h_addr_list[i], buf, _lib.INET6_ADDRSTRLEN):
                self.addresses.append(maybe_str(_ffi.string(buf, _lib.INET6_ADDRSTRLEN)))
            i += 1