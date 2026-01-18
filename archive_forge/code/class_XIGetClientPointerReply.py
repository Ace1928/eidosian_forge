import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIGetClientPointerReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.set, self.deviceid = unpacker.unpack('xx2x4xBxH20x')
        self.bufsize = unpacker.offset - base