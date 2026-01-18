import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class CursorNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.subtype, self.window, self.cursor_serial, self.timestamp, self.name = unpacker.unpack('xB2xIIII12x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 1))
        buf.write(struct.pack('=B2xIIII12x', self.subtype, self.window, self.cursor_serial, self.timestamp, self.name))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, subtype, window, cursor_serial, timestamp, name):
        self = cls.__new__(cls)
        self.subtype = subtype
        self.window = window
        self.cursor_serial = cursor_serial
        self.timestamp = timestamp
        self.name = name
        return self