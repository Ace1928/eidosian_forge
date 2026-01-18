import xcffib
import struct
import io
from . import xproto
class GetTexEnvfvReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.n, self.datum = unpacker.unpack('xx2x4x4xIf12x')
        self.data = xcffib.List(unpacker, 'f', self.n)
        self.bufsize = unpacker.offset - base