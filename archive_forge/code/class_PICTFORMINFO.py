import xcffib
import struct
import io
from . import xproto
class PICTFORMINFO(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.id, self.type, self.depth = unpacker.unpack('IBB2x')
        self.direct = DIRECTFORMAT(unpacker)
        self.colormap, = unpacker.unpack('I')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IBB2x', self.id, self.type, self.depth))
        buf.write(self.direct.pack() if hasattr(self.direct, 'pack') else DIRECTFORMAT.synthetic(*self.direct).pack())
        buf.write(struct.pack('=I', self.colormap))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, id, type, depth, direct, colormap):
        self = cls.__new__(cls)
        self.id = id
        self.type = type
        self.depth = depth
        self.direct = direct
        self.colormap = colormap
        return self