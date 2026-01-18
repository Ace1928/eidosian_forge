import xcffib
import struct
import io
from . import xproto
from . import render
class GetProviderInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.timestamp, self.capabilities, self.num_crtcs, self.num_outputs, self.num_associated_providers, self.name_len = unpacker.unpack('xB2x4xIIHHHH8x')
        self.crtcs = xcffib.List(unpacker, 'I', self.num_crtcs)
        unpacker.pad('I')
        self.outputs = xcffib.List(unpacker, 'I', self.num_outputs)
        unpacker.pad('I')
        self.associated_providers = xcffib.List(unpacker, 'I', self.num_associated_providers)
        unpacker.pad('I')
        self.associated_capability = xcffib.List(unpacker, 'I', self.num_associated_providers)
        unpacker.pad('c')
        self.name = xcffib.List(unpacker, 'c', self.name_len)
        self.bufsize = unpacker.offset - base