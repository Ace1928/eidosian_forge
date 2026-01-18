from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def _init_x(self):
    if core is None:
        raise XcffibException('No core protocol object has been set.  Did you import xcffib.xproto?')
    self.core = core(self)
    self.setup = self.get_setup()
    self._event_offsets = OffsetMap(core_events)
    self._error_offsets = OffsetMap(core_errors)
    self._setup_extensions()