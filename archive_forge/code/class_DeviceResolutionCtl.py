import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceResolutionCtl(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.control_id, self.len, self.first_valuator, self.num_valuators = unpacker.unpack('HHBB2x')
        self.resolution_values = xcffib.List(unpacker, 'I', self.num_valuators)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHBB2x', self.control_id, self.len, self.first_valuator, self.num_valuators))
        buf.write(xcffib.pack_list(self.resolution_values, 'I'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, control_id, len, first_valuator, num_valuators, resolution_values):
        self = cls.__new__(cls)
        self.control_id = control_id
        self.len = len
        self.first_valuator = first_valuator
        self.num_valuators = num_valuators
        self.resolution_values = resolution_values
        return self