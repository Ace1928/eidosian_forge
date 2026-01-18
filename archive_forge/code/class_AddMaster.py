import xcffib
import struct
import io
from . import xfixes
from . import xproto
class AddMaster(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.name_len, self.send_core, self.enable = unpacker.unpack('HHHBB')
        self.name = xcffib.List(unpacker, 'c', self.name_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHBB', self.type, self.len, self.name_len, self.send_core, self.enable))
        buf.write(xcffib.pack_list(self.name, 'c'))
        buf.write(struct.pack('=4x'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, type, len, name_len, send_core, enable, name):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.name_len = name_len
        self.send_core = send_core
        self.enable = enable
        self.name = name
        return self