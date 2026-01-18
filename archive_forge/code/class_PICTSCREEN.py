import xcffib
import struct
import io
from . import xproto
class PICTSCREEN(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.num_depths, self.fallback = unpacker.unpack('II')
        self.depths = xcffib.List(unpacker, PICTDEPTH, self.num_depths)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.num_depths, self.fallback))
        buf.write(xcffib.pack_list(self.depths, PICTDEPTH))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, num_depths, fallback, depths):
        self = cls.__new__(cls)
        self.num_depths = num_depths
        self.fallback = fallback
        self.depths = depths
        return self