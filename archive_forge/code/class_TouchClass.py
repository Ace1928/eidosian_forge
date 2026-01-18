import xcffib
import struct
import io
from . import xfixes
from . import xproto
class TouchClass(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.sourceid, self.mode, self.num_touches = unpacker.unpack('HHHBB')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHBB', self.type, self.len, self.sourceid, self.mode, self.num_touches))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, type, len, sourceid, mode, num_touches):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.sourceid = sourceid
        self.mode = mode
        self.num_touches = num_touches
        return self