import xcffib
import struct
import io
from . import xproto
from . import randr
from . import xfixes
from . import sync
class QueryCapabilitiesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.capabilities, = unpacker.unpack('xx2x4xI')
        self.bufsize = unpacker.offset - base