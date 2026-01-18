import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceTimeCoord(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.time, = unpacker.unpack('I')
        self.axisvalues = xcffib.List(unpacker, 'i', self.num_axes)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=I', self.time))
        buf.write(xcffib.pack_list(self.axisvalues, 'i'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, time, axisvalues):
        self = cls.__new__(cls)
        self.time = time
        self.axisvalues = axisvalues
        return self