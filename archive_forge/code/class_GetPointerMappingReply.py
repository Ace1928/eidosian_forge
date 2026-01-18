import xcffib
import struct
import io
class GetPointerMappingReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.map_len, = unpacker.unpack('xB2x4x24x')
        self.map = xcffib.List(unpacker, 'B', self.map_len)
        self.bufsize = unpacker.offset - base