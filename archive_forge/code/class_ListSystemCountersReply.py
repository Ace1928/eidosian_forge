import xcffib
import struct
import io
from . import xproto
class ListSystemCountersReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.counters_len, = unpacker.unpack('xx2x4xI20x')
        self.counters = xcffib.List(unpacker, SYSTEMCOUNTER, self.counters_len)
        self.bufsize = unpacker.offset - base