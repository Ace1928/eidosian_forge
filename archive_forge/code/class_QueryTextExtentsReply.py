import xcffib
import struct
import io
class QueryTextExtentsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.draw_direction, self.font_ascent, self.font_descent, self.overall_ascent, self.overall_descent, self.overall_width, self.overall_left, self.overall_right = unpacker.unpack('xB2x4xhhhhiii')
        self.bufsize = unpacker.offset - base