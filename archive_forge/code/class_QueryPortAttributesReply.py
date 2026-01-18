import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryPortAttributesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_attributes, self.text_size = unpacker.unpack('xx2x4xII16x')
        self.attributes = xcffib.List(unpacker, AttributeInfo, self.num_attributes)
        self.bufsize = unpacker.offset - base