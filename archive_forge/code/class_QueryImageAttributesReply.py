import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryImageAttributesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_planes, self.data_size, self.width, self.height = unpacker.unpack('xx2x4xIIHH12x')
        self.pitches = xcffib.List(unpacker, 'I', self.num_planes)
        unpacker.pad('I')
        self.offsets = xcffib.List(unpacker, 'I', self.num_planes)
        self.bufsize = unpacker.offset - base