import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ScrollClass(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.sourceid, self.number, self.scroll_type, self.flags = unpacker.unpack('HHHHH2xI')
        self.increment = FP3232(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHHH2xI', self.type, self.len, self.sourceid, self.number, self.scroll_type, self.flags))
        buf.write(self.increment.pack() if hasattr(self.increment, 'pack') else FP3232.synthetic(*self.increment).pack())
        return buf.getvalue()

    @classmethod
    def synthetic(cls, type, len, sourceid, number, scroll_type, flags, increment):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.sourceid = sourceid
        self.number = number
        self.scroll_type = scroll_type
        self.flags = flags
        self.increment = increment
        return self