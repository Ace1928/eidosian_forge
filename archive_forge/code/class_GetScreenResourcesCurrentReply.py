import xcffib
import struct
import io
from . import xproto
from . import render
class GetScreenResourcesCurrentReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.config_timestamp, self.num_crtcs, self.num_outputs, self.num_modes, self.names_len = unpacker.unpack('xx2x4xIIHHHH8x')
        self.crtcs = xcffib.List(unpacker, 'I', self.num_crtcs)
        unpacker.pad('I')
        self.outputs = xcffib.List(unpacker, 'I', self.num_outputs)
        unpacker.pad(ModeInfo)
        self.modes = xcffib.List(unpacker, ModeInfo, self.num_modes)
        unpacker.pad('B')
        self.names = xcffib.List(unpacker, 'B', self.names_len)
        self.bufsize = unpacker.offset - base