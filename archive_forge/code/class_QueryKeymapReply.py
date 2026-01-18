import xcffib
import struct
import io
class QueryKeymapReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x')
        self.keys = xcffib.List(unpacker, 'B', 32)
        self.bufsize = unpacker.offset - base