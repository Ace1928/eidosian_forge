import xcffib
import struct
import io
class GetGammaReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.red, self.green, self.blue = unpacker.unpack('xx2x4xIII12x')
        self.bufsize = unpacker.offset - base