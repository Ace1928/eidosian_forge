import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ValuatorClass(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.sourceid, self.number, self.label = unpacker.unpack('HHHHI')
        self.min = FP3232(unpacker)
        unpacker.pad(FP3232)
        self.max = FP3232(unpacker)
        unpacker.pad(FP3232)
        self.value = FP3232(unpacker)
        self.resolution, self.mode = unpacker.unpack('IB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHHI', self.type, self.len, self.sourceid, self.number, self.label))
        buf.write(self.min.pack() if hasattr(self.min, 'pack') else FP3232.synthetic(*self.min).pack())
        buf.write(self.max.pack() if hasattr(self.max, 'pack') else FP3232.synthetic(*self.max).pack())
        buf.write(self.value.pack() if hasattr(self.value, 'pack') else FP3232.synthetic(*self.value).pack())
        buf.write(struct.pack('=I', self.resolution))
        buf.write(struct.pack('=B', self.mode))
        buf.write(struct.pack('=3x'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, type, len, sourceid, number, label, min, max, value, resolution, mode):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.sourceid = sourceid
        self.number = number
        self.label = label
        self.min = min
        self.max = max
        self.value = value
        self.resolution = resolution
        self.mode = mode
        return self