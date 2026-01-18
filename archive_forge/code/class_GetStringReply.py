import xcffib
import struct
import io
from . import xproto
class GetStringReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.n, = unpacker.unpack('xx2x4x4xI16x')
        self.string = xcffib.List(unpacker, 'c', self.n)
        self.bufsize = unpacker.offset - base