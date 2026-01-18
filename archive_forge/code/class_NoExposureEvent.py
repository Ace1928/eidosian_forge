import xcffib
import struct
import io
class NoExposureEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.drawable, self.minor_opcode, self.major_opcode = unpacker.unpack('xx2xIHBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 14))
        buf.write(struct.pack('=x2xIHBx', self.drawable, self.minor_opcode, self.major_opcode))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, drawable, minor_opcode, major_opcode):
        self = cls.__new__(cls)
        self.drawable = drawable
        self.minor_opcode = minor_opcode
        self.major_opcode = major_opcode
        return self