import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceModifierMappingReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.keycodes_per_modifier = unpacker.unpack('xB2x4xB23x')
        self.keymaps = xcffib.List(unpacker, 'B', self.keycodes_per_modifier * 8)
        self.bufsize = unpacker.offset - base