import xcffib
import struct
import io
from . import xfixes
from . import xproto
class AttachSlave(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.deviceid, self.master = unpacker.unpack('HHHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHH', self.type, self.len, self.deviceid, self.master))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, type, len, deviceid, master):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.deviceid = deviceid
        self.master = master
        return self