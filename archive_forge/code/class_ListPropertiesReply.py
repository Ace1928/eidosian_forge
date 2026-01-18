import xcffib
import struct
import io
class ListPropertiesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.atoms_len, = unpacker.unpack('xx2x4xH22x')
        self.atoms = xcffib.List(unpacker, 'I', self.atoms_len)
        self.bufsize = unpacker.offset - base