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
class ares_query_srv_result(AresResult):
    __slots__ = ('host', 'port', 'priority', 'weight', 'ttl')
    type = 'SRV'

    def __init__(self, srv):
        self.host = maybe_str(_ffi.string(srv.host))
        self.port = srv.port
        self.priority = srv.priority
        self.weight = srv.weight
        self.ttl = -1