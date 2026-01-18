import xcffib
import struct
import io
from . import xproto
class ClientIdValue(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.spec = ClientIdSpec(unpacker)
        self.length, = unpacker.unpack('I')
        unpacker.pad('I')
        self.value = xcffib.List(unpacker, 'I', self.length // 4)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.spec.pack() if hasattr(self.spec, 'pack') else ClientIdSpec.synthetic(*self.spec).pack())
        buf.write(struct.pack('=I', self.length))
        buf.write(xcffib.pack_list(self.value, 'I'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, spec, length, value):
        self = cls.__new__(cls)
        self.spec = spec
        self.length = length
        self.value = value
        return self