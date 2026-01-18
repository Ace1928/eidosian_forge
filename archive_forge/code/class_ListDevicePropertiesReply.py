import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ListDevicePropertiesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.num_atoms = unpacker.unpack('xB2x4xH22x')
        self.atoms = xcffib.List(unpacker, 'I', self.num_atoms)
        self.bufsize = unpacker.offset - base