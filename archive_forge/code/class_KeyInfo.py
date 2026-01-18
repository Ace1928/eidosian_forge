import xcffib
import struct
import io
from . import xfixes
from . import xproto
class KeyInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len, self.min_keycode, self.max_keycode, self.num_keys = unpacker.unpack('BBBBH2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBBBH2x', self.class_id, self.len, self.min_keycode, self.max_keycode, self.num_keys))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, class_id, len, min_keycode, max_keycode, num_keys):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.min_keycode = min_keycode
        self.max_keycode = max_keycode
        self.num_keys = num_keys
        return self