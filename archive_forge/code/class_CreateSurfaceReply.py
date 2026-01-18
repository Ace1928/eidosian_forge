import xcffib
import struct
import io
from . import xv
class CreateSurfaceReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x24x')
        self.priv_data = xcffib.List(unpacker, 'I', self.length)
        self.bufsize = unpacker.offset - base