import xcffib
import struct
import io
from . import xfixes
from . import xproto
class InputClassInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.event_type_base = unpacker.unpack('BB')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BB', self.class_id, self.event_type_base))
        return buf.getvalue()
    fixed_size = 2

    @classmethod
    def synthetic(cls, class_id, event_type_base):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.event_type_base = event_type_base
        return self