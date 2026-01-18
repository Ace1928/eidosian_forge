import xcffib
import struct
import io
from . import xproto
class GetParamReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.is_param_recognized, self.value_hi, self.value_lo = unpacker.unpack('xB2x4xII')
        self.bufsize = unpacker.offset - base