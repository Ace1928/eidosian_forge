import xcffib
import struct
import io
from . import xproto
class GetScreenSizeReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.width, self.height, self.window, self.screen = unpacker.unpack('xx2x4xIIII')
        self.bufsize = unpacker.offset - base