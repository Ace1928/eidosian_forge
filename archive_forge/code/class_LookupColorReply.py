import xcffib
import struct
import io
class LookupColorReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.exact_red, self.exact_green, self.exact_blue, self.visual_red, self.visual_green, self.visual_blue = unpacker.unpack('xx2x4xHHHHHH')
        self.bufsize = unpacker.offset - base