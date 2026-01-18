from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class ExtensionKey(object):
    """ This definitely isn't needed, but we keep it around for compatibility
    with xpyb.
    """

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, o):
        return self.name == o.name

    def __ne__(self, o):
        return self.name != o.name

    def to_cffi(self):
        c_key = ffi.new('struct xcb_extension_t *')
        c_key.name = name = ffi.new('char[]', self.name.encode())
        cffi_explicit_lifetimes[c_key] = name
        c_key.global_id = 0
        return c_key