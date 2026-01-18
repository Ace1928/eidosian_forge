import xcffib
import struct
import io
from . import xfixes
from . import xproto
class KeyClass(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.sourceid, self.num_keys = unpacker.unpack('HHHH')
        self.keys = xcffib.List(unpacker, 'I', self.num_keys)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHH', self.type, self.len, self.sourceid, self.num_keys))
        buf.write(xcffib.pack_list(self.keys, 'I'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, type, len, sourceid, num_keys, keys):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.sourceid = sourceid
        self.num_keys = num_keys
        self.keys = keys
        return self