import xcffib
import struct
import io
class ClientInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.client_resource, self.num_ranges = unpacker.unpack('II')
        self.ranges = xcffib.List(unpacker, Range, self.num_ranges)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.client_resource, self.num_ranges))
        buf.write(xcffib.pack_list(self.ranges, Range))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, client_resource, num_ranges, ranges):
        self = cls.__new__(cls)
        self.client_resource = client_resource
        self.num_ranges = num_ranges
        self.ranges = ranges
        return self