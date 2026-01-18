import xcffib
import struct
import io
from . import xproto
from . import shm
class ListImageFormatsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_formats, = unpacker.unpack('xx2x4xI20x')
        self.format = xcffib.List(unpacker, ImageFormatInfo, self.num_formats)
        self.bufsize = unpacker.offset - base