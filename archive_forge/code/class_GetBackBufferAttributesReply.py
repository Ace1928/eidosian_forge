import xcffib
import struct
import io
from . import xproto
class GetBackBufferAttributesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x')
        self.attributes = BufferAttributes(unpacker)
        unpacker.unpack('20x')
        self.bufsize = unpacker.offset - base