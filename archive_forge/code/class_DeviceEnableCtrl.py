import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceEnableCtrl(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.control_id, self.len, self.enable = unpacker.unpack('HHB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHB3x', self.control_id, self.len, self.enable))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, control_id, len, enable):
        self = cls.__new__(cls)
        self.control_id = control_id
        self.len = len
        self.enable = enable
        return self