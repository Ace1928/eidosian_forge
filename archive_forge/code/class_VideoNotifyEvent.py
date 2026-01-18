import xcffib
import struct
import io
from . import xproto
from . import shm
class VideoNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.reason, self.time, self.drawable, self.port = unpacker.unpack('xB2xIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=B2xIII', self.reason, self.time, self.drawable, self.port))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, reason, time, drawable, port):
        self = cls.__new__(cls)
        self.reason = reason
        self.time = time
        self.drawable = drawable
        self.port = port
        return self