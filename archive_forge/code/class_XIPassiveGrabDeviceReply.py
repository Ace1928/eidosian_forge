import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIPassiveGrabDeviceReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_modifiers, = unpacker.unpack('xx2x4xH22x')
        self.modifiers = xcffib.List(unpacker, GrabModifierInfo, self.num_modifiers)
        self.bufsize = unpacker.offset - base