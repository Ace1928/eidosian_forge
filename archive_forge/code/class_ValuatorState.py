import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ValuatorState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len, self.num_valuators, self.mode = unpacker.unpack('BBBB')
        self.valuators = xcffib.List(unpacker, 'i', self.num_valuators)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBBB', self.class_id, self.len, self.num_valuators, self.mode))
        buf.write(xcffib.pack_list(self.valuators, 'i'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, class_id, len, num_valuators, mode, valuators):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.num_valuators = num_valuators
        self.mode = mode
        self.valuators = valuators
        return self