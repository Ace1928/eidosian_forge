import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.device_type, self.device_id, self.num_class_info, self.device_use = unpacker.unpack('IBBBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IBBBx', self.device_type, self.device_id, self.num_class_info, self.device_use))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, device_type, device_id, num_class_info, device_use):
        self = cls.__new__(cls)
        self.device_type = device_type
        self.device_id = device_id
        self.num_class_info = num_class_info
        self.device_use = device_use
        return self