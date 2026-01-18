import xcffib
import struct
import io
from . import xfixes
from . import xproto
class HierarchyEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.flags, self.num_infos = unpacker.unpack('xx2xHIIH10x')
        self.infos = xcffib.List(unpacker, HierarchyInfo, self.num_infos)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 11))
        buf.write(struct.pack('=x2xHIIH10x', self.deviceid, self.time, self.flags, self.num_infos))
        buf.write(xcffib.pack_list(self.infos, HierarchyInfo))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, flags, num_infos, infos):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.flags = flags
        self.num_infos = num_infos
        self.infos = infos
        return self