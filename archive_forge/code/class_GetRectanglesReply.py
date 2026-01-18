import xcffib
import struct
import io
from . import xproto
class GetRectanglesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.ordering, self.rectangles_len = unpacker.unpack('xB2x4xI20x')
        self.rectangles = xcffib.List(unpacker, xproto.RECTANGLE, self.rectangles_len)
        self.bufsize = unpacker.offset - base