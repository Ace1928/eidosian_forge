import xcffib
import struct
import io
from . import xproto
class WaitMSCReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.ust_hi, self.ust_lo, self.msc_hi, self.msc_lo, self.sbc_hi, self.sbc_lo = unpacker.unpack('xx2x4xIIIIII')
        self.bufsize = unpacker.offset - base