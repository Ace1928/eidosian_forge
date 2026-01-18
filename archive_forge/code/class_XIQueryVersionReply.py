import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIQueryVersionReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.major_version, self.minor_version = unpacker.unpack('xx2x4xHH20x')
        self.bufsize = unpacker.offset - base