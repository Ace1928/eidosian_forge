import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDevicePropertyReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.type, self.bytes_after, self.num_items, self.format, self.device_id = unpacker.unpack('xB2x4xIIIBB10x')
        if self.format & PropertyFormat._8Bits:
            self.data8 = xcffib.List(unpacker, 'B', self.num_items)
        if self.format & PropertyFormat._16Bits:
            self.data16 = xcffib.List(unpacker, 'H', self.num_items)
        if self.format & PropertyFormat._32Bits:
            self.data32 = xcffib.List(unpacker, 'I', self.num_items)
        self.bufsize = unpacker.offset - base