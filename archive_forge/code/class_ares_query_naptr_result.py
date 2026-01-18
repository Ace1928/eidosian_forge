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
class ares_query_naptr_result(AresResult):
    __slots__ = ('order', 'preference', 'flags', 'service', 'regex', 'replacement', 'ttl')
    type = 'NAPTR'

    def __init__(self, naptr):
        self.order = naptr.order
        self.preference = naptr.preference
        self.flags = maybe_str(_ffi.string(naptr.flags))
        self.service = maybe_str(_ffi.string(naptr.service))
        self.regex = maybe_str(_ffi.string(naptr.regexp))
        self.replacement = maybe_str(_ffi.string(naptr.replacement))
        self.ttl = -1