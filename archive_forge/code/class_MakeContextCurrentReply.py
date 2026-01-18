import xcffib
import struct
import io
from . import xproto
class MakeContextCurrentReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.context_tag, = unpacker.unpack('xx2x4xI20x')
        self.bufsize = unpacker.offset - base