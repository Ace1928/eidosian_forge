import xcffib
import struct
import io
class GetXIDListReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.ids_len, = unpacker.unpack('xx2x4xI20x')
        self.ids = xcffib.List(unpacker, 'I', self.ids_len)
        self.bufsize = unpacker.offset - base