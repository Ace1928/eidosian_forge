import xcffib
import struct
import io
from . import xproto
class TRANSFORM(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.matrix11, self.matrix12, self.matrix13, self.matrix21, self.matrix22, self.matrix23, self.matrix31, self.matrix32, self.matrix33 = unpacker.unpack('iiiiiiiii')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=iiiiiiiii', self.matrix11, self.matrix12, self.matrix13, self.matrix21, self.matrix22, self.matrix23, self.matrix31, self.matrix32, self.matrix33))
        return buf.getvalue()
    fixed_size = 36

    @classmethod
    def synthetic(cls, matrix11, matrix12, matrix13, matrix21, matrix22, matrix23, matrix31, matrix32, matrix33):
        self = cls.__new__(cls)
        self.matrix11 = matrix11
        self.matrix12 = matrix12
        self.matrix13 = matrix13
        self.matrix21 = matrix21
        self.matrix22 = matrix22
        self.matrix23 = matrix23
        self.matrix31 = matrix31
        self.matrix32 = matrix32
        self.matrix33 = matrix33
        return self