import xcffib
import struct
import io
from . import xv
class ListSurfaceTypesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num, = unpacker.unpack('xx2x4xI20x')
        self.surfaces = xcffib.List(unpacker, SurfaceInfo, self.num)
        self.bufsize = unpacker.offset - base