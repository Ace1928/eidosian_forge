import xcffib
import struct
import io
class GraphicsExposureEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.drawable, self.x, self.y, self.width, self.height, self.minor_opcode, self.count, self.major_opcode = unpacker.unpack('xx2xIHHHHHHB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 13))
        buf.write(struct.pack('=x2xIHHHHHHB3x', self.drawable, self.x, self.y, self.width, self.height, self.minor_opcode, self.count, self.major_opcode))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, drawable, x, y, width, height, minor_opcode, count, major_opcode):
        self = cls.__new__(cls)
        self.drawable = drawable
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.minor_opcode = minor_opcode
        self.count = count
        self.major_opcode = major_opcode
        return self