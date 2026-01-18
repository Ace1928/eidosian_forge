import xcffib
import struct
import io
from . import xproto
from . import randr
from . import xfixes
from . import sync
class GenericEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.extension, self.length, self.evtype, self.event = unpacker.unpack('xB2xIH2xI')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=B2xIH2xI', self.extension, self.length, self.evtype, self.event))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, extension, length, evtype, event):
        self = cls.__new__(cls)
        self.extension = extension
        self.length = length
        self.evtype = evtype
        self.event = event
        return self