import xcffib
import struct
import io
from . import xproto
from . import render
class GetCrtcInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.timestamp, self.x, self.y, self.width, self.height, self.mode, self.rotation, self.rotations, self.num_outputs, self.num_possible_outputs = unpacker.unpack('xB2x4xIhhHHIHHHH')
        self.outputs = xcffib.List(unpacker, 'I', self.num_outputs)
        unpacker.pad('I')
        self.possible = xcffib.List(unpacker, 'I', self.num_possible_outputs)
        self.bufsize = unpacker.offset - base