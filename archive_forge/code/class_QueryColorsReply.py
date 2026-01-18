import xcffib
import struct
import io
class QueryColorsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.colors_len, = unpacker.unpack('xx2x4xH22x')
        self.colors = xcffib.List(unpacker, RGB, self.colors_len)
        self.bufsize = unpacker.offset - base