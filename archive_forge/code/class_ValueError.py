import xcffib
import struct
import io
class ValueError(xcffib.Error):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Error.__init__(self, unpacker)
        base = unpacker.offset
        self.bad_value, self.minor_opcode, self.major_opcode = unpacker.unpack('xx2xIHBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 2))
        buf.write(struct.pack('=x2xIHBx', self.bad_value, self.minor_opcode, self.major_opcode))
        return buf.getvalue()