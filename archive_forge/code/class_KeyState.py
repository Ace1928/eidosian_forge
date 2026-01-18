import xcffib
import struct
import io
from . import xfixes
from . import xproto
class KeyState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len, self.num_keys = unpacker.unpack('BBBx')
        self.keys = xcffib.List(unpacker, 'B', 32)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBBx', self.class_id, self.len, self.num_keys))
        buf.write(xcffib.pack_list(self.keys, 'B'))
        return buf.getvalue()
    fixed_size = 36

    @classmethod
    def synthetic(cls, class_id, len, num_keys, keys):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.num_keys = num_keys
        self.keys = keys
        return self