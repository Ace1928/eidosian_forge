import xcffib
import struct
import io
from . import xproto
class OpenReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.nfd, = unpacker.unpack('xB2x4x24x')
        self.bufsize = unpacker.offset - base