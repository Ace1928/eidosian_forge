from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class Protobj(object):
    """ Note: Unlike xcb.Protobj, this does NOT implement the sequence
    protocol. I found this behavior confusing: Protobj would implement the
    sequence protocol on self.buf, and then List would go and implement it on
    List.

    Instead, when we need to create a new event from an existing event, we
    repack that event into a MemoryUnpacker and use that instead (see
    eventToUnpacker in the generator for more info.)
    """

    def __init__(self, unpacker):
        """
        Params:
        - unpacker: an Unpacker object
        """
        if unpacker.known_max is not None:
            self.bufsize = unpacker.known_max