import xcffib
import struct
import io
from . import xproto
class QueryFiltersReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_aliases, self.num_filters = unpacker.unpack('xx2x4xII16x')
        self.aliases = xcffib.List(unpacker, 'H', self.num_aliases)
        unpacker.pad(xproto.STR)
        self.filters = xcffib.List(unpacker, xproto.STR, self.num_filters)
        self.bufsize = unpacker.offset - base