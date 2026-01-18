import xcffib
import struct
import io
from . import xproto
class InvalidateBuffersEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.drawable, = unpacker.unpack('xx2xI')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 1))
        buf.write(struct.pack('=x2xI', self.drawable))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, drawable):
        self = cls.__new__(cls)
        self.drawable = drawable
        return self