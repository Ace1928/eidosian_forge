import xcffib
import struct
import io
class GetGeometryReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.depth, self.root, self.x, self.y, self.width, self.height, self.border_width = unpacker.unpack('xB2x4xIhhHHH2x')
        self.bufsize = unpacker.offset - base