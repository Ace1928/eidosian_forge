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
class ares_query_caa_result(AresResult):
    __slots__ = ('critical', 'property', 'value', 'ttl')
    type = 'CAA'

    def __init__(self, caa):
        self.critical = caa.critical
        self.property = maybe_str(_ffi.string(caa.property, caa.plength))
        self.value = maybe_str(_ffi.string(caa.value, caa.length))
        self.ttl = -1