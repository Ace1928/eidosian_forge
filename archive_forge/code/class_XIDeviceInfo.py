import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIDeviceInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.type, self.attachment, self.num_classes, self.name_len, self.enabled = unpacker.unpack('HHHHHBx')
        self.name = xcffib.List(unpacker, 'c', self.name_len)
        unpacker.pad(DeviceClass)
        self.classes = xcffib.List(unpacker, DeviceClass, self.num_classes)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHHHBx', self.deviceid, self.type, self.attachment, self.num_classes, self.name_len, self.enabled))
        buf.write(xcffib.pack_list(self.name, 'c'))
        buf.write(struct.pack('=4x'))
        buf.write(xcffib.pack_list(self.classes, DeviceClass))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, type, attachment, num_classes, name_len, enabled, name, classes):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.type = type
        self.attachment = attachment
        self.num_classes = num_classes
        self.name_len = name_len
        self.enabled = enabled
        self.name = name
        self.classes = classes
        return self