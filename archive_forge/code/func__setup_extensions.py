from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def _setup_extensions(self):
    for key, (_, events, errors) in extensions.items():
        c_ext = key.to_cffi()
        reply = lib.xcb_get_extension_data(self._conn, c_ext)
        self._event_offsets.add(reply.first_event, reply.major_opcode, events)
        self._error_offsets.add(reply.first_error, reply.major_opcode, errors)