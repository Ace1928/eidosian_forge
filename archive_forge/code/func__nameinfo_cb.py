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
@_ffi.def_extern()
def _nameinfo_cb(arg, status, timeouts, node, service):
    callback = _ffi.from_handle(arg)
    _global_set.discard(arg)
    if status != _lib.ARES_SUCCESS:
        result = None
    else:
        result = ares_nameinfo_result(node, service)
        status = None
    callback(result, status)