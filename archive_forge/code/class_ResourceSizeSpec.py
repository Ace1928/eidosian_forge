import xcffib
import struct
import io
from . import xproto
class ResourceSizeSpec(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.spec = ResourceIdSpec(unpacker)
        self.bytes, self.ref_count, self.use_count = unpacker.unpack('III')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.spec.pack() if hasattr(self.spec, 'pack') else ResourceIdSpec.synthetic(*self.spec).pack())
        buf.write(struct.pack('=I', self.bytes))
        buf.write(struct.pack('=I', self.ref_count))
        buf.write(struct.pack('=I', self.use_count))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, spec, bytes, ref_count, use_count):
        self = cls.__new__(cls)
        self.spec = spec
        self.bytes = bytes
        self.ref_count = ref_count
        self.use_count = use_count
        return self