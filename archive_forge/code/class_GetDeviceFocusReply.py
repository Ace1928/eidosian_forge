import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceFocusReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.focus, self.time, self.revert_to = unpacker.unpack('xB2x4xIIB15x')
        self.bufsize = unpacker.offset - base