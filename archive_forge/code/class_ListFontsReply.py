import xcffib
import struct
import io
class ListFontsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.names_len, = unpacker.unpack('xx2x4xH22x')
        self.names = xcffib.List(unpacker, STR, self.names_len)
        self.bufsize = unpacker.offset - base