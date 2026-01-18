import xcffib
import struct
import io
from . import xfixes
from . import xproto
class HierarchyInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.attachment, self.type, self.enabled, self.flags = unpacker.unpack('HHBB2xI')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHBB2xI', self.deviceid, self.attachment, self.type, self.enabled, self.flags))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, deviceid, attachment, type, enabled, flags):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.attachment = attachment
        self.type = type
        self.enabled = enabled
        self.flags = flags
        return self