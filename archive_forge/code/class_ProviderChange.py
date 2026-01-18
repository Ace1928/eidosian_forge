import xcffib
import struct
import io
from . import xproto
from . import render
class ProviderChange(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.window, self.provider = unpacker.unpack('III16x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=III16x', self.timestamp, self.window, self.provider))
        return buf.getvalue()
    fixed_size = 28

    @classmethod
    def synthetic(cls, timestamp, window, provider):
        self = cls.__new__(cls)
        self.timestamp = timestamp
        self.window = window
        self.provider = provider
        return self