import xcffib
import struct
import io
from . import xproto
from . import render
class GetCrtcGammaReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.size, = unpacker.unpack('xx2x4xH22x')
        self.red = xcffib.List(unpacker, 'H', self.size)
        unpacker.pad('H')
        self.green = xcffib.List(unpacker, 'H', self.size)
        unpacker.pad('H')
        self.blue = xcffib.List(unpacker, 'H', self.size)
        self.bufsize = unpacker.offset - base