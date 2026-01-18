import xcffib
import struct
import io
from . import xproto
class DIRECTFORMAT(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.red_shift, self.red_mask, self.green_shift, self.green_mask, self.blue_shift, self.blue_mask, self.alpha_shift, self.alpha_mask = unpacker.unpack('HHHHHHHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHHHHHH', self.red_shift, self.red_mask, self.green_shift, self.green_mask, self.blue_shift, self.blue_mask, self.alpha_shift, self.alpha_mask))
        return buf.getvalue()
    fixed_size = 16

    @classmethod
    def synthetic(cls, red_shift, red_mask, green_shift, green_mask, blue_shift, blue_mask, alpha_shift, alpha_mask):
        self = cls.__new__(cls)
        self.red_shift = red_shift
        self.red_mask = red_mask
        self.green_shift = green_shift
        self.green_mask = green_mask
        self.blue_shift = blue_shift
        self.blue_mask = blue_mask
        self.alpha_shift = alpha_shift
        self.alpha_mask = alpha_mask
        return self