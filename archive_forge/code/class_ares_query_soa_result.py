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
class ares_query_soa_result(AresResult):
    __slots__ = ('nsname', 'hostmaster', 'serial', 'refresh', 'retry', 'expires', 'minttl', 'ttl')
    type = 'SOA'

    def __init__(self, soa):
        self.nsname = maybe_str(_ffi.string(soa.nsname))
        self.hostmaster = maybe_str(_ffi.string(soa.hostmaster))
        self.serial = soa.serial
        self.refresh = soa.refresh
        self.retry = soa.retry
        self.expires = soa.expire
        self.minttl = soa.minttl
        self.ttl = -1