import xcffib
import struct
import io
from . import xproto
class QueryServerStringReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.str_len, = unpacker.unpack('xx2x4x4xI16x')
        self.string = xcffib.List(unpacker, 'c', self.str_len)
        self.bufsize = unpacker.offset - base