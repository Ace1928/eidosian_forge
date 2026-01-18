import xcffib
import struct
import io
class COLORITEM(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.pixel, self.red, self.green, self.blue, self.flags = unpacker.unpack('IHHHBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IHHHBx', self.pixel, self.red, self.green, self.blue, self.flags))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, pixel, red, green, blue, flags):
        self = cls.__new__(cls)
        self.pixel = pixel
        self.red = red
        self.green = green
        self.blue = blue
        self.flags = flags
        return self