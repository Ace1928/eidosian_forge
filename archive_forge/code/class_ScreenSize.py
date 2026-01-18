import xcffib
import struct
import io
from . import xproto
from . import render
class ScreenSize(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.width, self.height, self.mwidth, self.mheight = unpacker.unpack('HHHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHH', self.width, self.height, self.mwidth, self.mheight))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, width, height, mwidth, mheight):
        self = cls.__new__(cls)
        self.width = width
        self.height = height
        self.mwidth = mwidth
        self.mheight = mheight
        return self