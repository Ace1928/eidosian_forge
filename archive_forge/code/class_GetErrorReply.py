import xcffib
import struct
import io
from . import xproto
class GetErrorReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.error, = unpacker.unpack('xx2x4xi')
        self.bufsize = unpacker.offset - base