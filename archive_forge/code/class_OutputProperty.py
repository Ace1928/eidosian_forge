import xcffib
import struct
import io
from . import xproto
from . import render
class OutputProperty(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.window, self.output, self.atom, self.timestamp, self.status = unpacker.unpack('IIIIB11x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IIIIB11x', self.window, self.output, self.atom, self.timestamp, self.status))
        return buf.getvalue()
    fixed_size = 28

    @classmethod
    def synthetic(cls, window, output, atom, timestamp, status):
        self = cls.__new__(cls)
        self.window = window
        self.output = output
        self.atom = atom
        self.timestamp = timestamp
        self.status = status
        return self