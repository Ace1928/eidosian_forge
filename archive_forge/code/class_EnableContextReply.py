import xcffib
import struct
import io
class EnableContextReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.category, self.element_header, self.client_swapped, self.xid_base, self.server_time, self.rec_sequence_num = unpacker.unpack('xB2x4xBB2xIII8x')
        self.data = xcffib.List(unpacker, 'B', self.length * 4)
        self.bufsize = unpacker.offset - base