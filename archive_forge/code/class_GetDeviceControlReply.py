import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceControlReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.status = unpacker.unpack('xB2x4xB23x')
        self.control = DeviceState(unpacker)
        self.bufsize = unpacker.offset - base