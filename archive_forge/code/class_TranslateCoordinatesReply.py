import xcffib
import struct
import io
class TranslateCoordinatesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.same_screen, self.child, self.dst_x, self.dst_y = unpacker.unpack('xB2x4xIhh')
        self.bufsize = unpacker.offset - base