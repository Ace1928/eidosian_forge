import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceKeyMappingReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.keysyms_per_keycode = unpacker.unpack('xB2x4xB23x')
        self.keysyms = xcffib.List(unpacker, 'I', self.length)
        self.bufsize = unpacker.offset - base