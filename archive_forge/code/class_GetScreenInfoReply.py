import xcffib
import struct
import io
from . import xproto
from . import render
class GetScreenInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.rotations, self.root, self.timestamp, self.config_timestamp, self.nSizes, self.sizeID, self.rotation, self.rate, self.nInfo = unpacker.unpack('xB2x4xIIIHHHHH2x')
        self.sizes = xcffib.List(unpacker, ScreenSize, self.nSizes)
        unpacker.pad(RefreshRates)
        self.rates = xcffib.List(unpacker, RefreshRates, self.nInfo - self.nSizes)
        self.bufsize = unpacker.offset - base