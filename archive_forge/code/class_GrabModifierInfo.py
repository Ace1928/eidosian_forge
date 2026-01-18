import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GrabModifierInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.modifiers, self.status = unpacker.unpack('IB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IB3x', self.modifiers, self.status))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, modifiers, status):
        self = cls.__new__(cls)
        self.modifiers = modifiers
        self.status = status
        return self