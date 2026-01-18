import xcffib
import struct
import io
from . import xproto
from . import shm
class GetPortAttributeReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.value, = unpacker.unpack('xx2x4xi')
        self.bufsize = unpacker.offset - base