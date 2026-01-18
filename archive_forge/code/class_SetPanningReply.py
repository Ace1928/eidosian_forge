import xcffib
import struct
import io
from . import xproto
from . import render
class SetPanningReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.timestamp = unpacker.unpack('xB2x4xI')
        self.bufsize = unpacker.offset - base