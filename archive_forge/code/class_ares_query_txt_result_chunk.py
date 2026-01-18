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
class ares_query_txt_result_chunk(AresResult):
    __slots__ = ('text', 'ttl')
    type = 'TXT'

    def __init__(self, txt):
        self.text = _ffi.string(txt.txt)
        self.ttl = -1