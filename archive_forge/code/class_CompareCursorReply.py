import xcffib
import struct
import io
from . import xproto
class CompareCursorReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.same, = unpacker.unpack('xB2x4x')
        self.bufsize = unpacker.offset - base