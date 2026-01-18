import xcffib
import struct
import io
from . import xproto
from . import render
class SetScreenConfigReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.new_timestamp, self.config_timestamp, self.root, self.subpixel_order = unpacker.unpack('xB2x4xIIIH10x')
        self.bufsize = unpacker.offset - base