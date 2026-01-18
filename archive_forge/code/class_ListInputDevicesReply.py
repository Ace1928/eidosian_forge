import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ListInputDevicesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.devices_len = unpacker.unpack('xB2x4xB23x')
        self.devices = xcffib.List(unpacker, DeviceInfo, self.devices_len)
        unpacker.pad(InputInfo)
        self.infos = xcffib.List(unpacker, InputInfo, sum(self.devices))
        unpacker.pad(xproto.STR)
        self.names = xcffib.List(unpacker, xproto.STR, self.devices_len)
        self.bufsize = unpacker.offset - base