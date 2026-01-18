import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetExtensionVersionReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.server_major, self.server_minor, self.present = unpacker.unpack('xB2x4xHHB19x')
        self.bufsize = unpacker.offset - base