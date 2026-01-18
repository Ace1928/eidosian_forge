import xcffib
import struct
import io
from . import xproto
from . import render
class ListOutputPropertiesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_atoms, = unpacker.unpack('xx2x4xH22x')
        self.atoms = xcffib.List(unpacker, 'I', self.num_atoms)
        self.bufsize = unpacker.offset - base