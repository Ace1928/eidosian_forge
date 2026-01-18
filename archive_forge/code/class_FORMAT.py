import xcffib
import struct
import io
class FORMAT(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.depth, self.bits_per_pixel, self.scanline_pad = unpacker.unpack('BBB5x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBB5x', self.depth, self.bits_per_pixel, self.scanline_pad))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, depth, bits_per_pixel, scanline_pad):
        self = cls.__new__(cls)
        self.depth = depth
        self.bits_per_pixel = bits_per_pixel
        self.scanline_pad = scanline_pad
        return self