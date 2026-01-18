import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class GetCursorImageReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.x, self.y, self.width, self.height, self.xhot, self.yhot, self.cursor_serial = unpacker.unpack('xx2x4xhhHHHHI8x')
        self.cursor_image = xcffib.List(unpacker, 'I', self.width * self.height)
        self.bufsize = unpacker.offset - base