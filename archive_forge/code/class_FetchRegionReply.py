import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class FetchRegionReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x')
        self.extents = xproto.RECTANGLE(unpacker)
        unpacker.unpack('16x')
        unpacker.pad(xproto.RECTANGLE)
        self.rectangles = xcffib.List(unpacker, xproto.RECTANGLE, self.length // 2)
        self.bufsize = unpacker.offset - base