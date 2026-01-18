import xcffib
import struct
import io
from . import xproto
class ClientIdSpec(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.client, self.mask = unpacker.unpack('II')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.client, self.mask))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, client, mask):
        self = cls.__new__(cls)
        self.client = client
        self.mask = mask
        return self