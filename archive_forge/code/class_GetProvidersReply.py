import xcffib
import struct
import io
from . import xproto
from . import render
class GetProvidersReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.num_providers = unpacker.unpack('xx2x4xIH18x')
        self.providers = xcffib.List(unpacker, 'I', self.num_providers)
        self.bufsize = unpacker.offset - base