import xcffib
import struct
import io
from . import xproto
class GetSeparableFilterReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.row_w, self.col_h = unpacker.unpack('xx2x4x8xii8x')
        self.rows_and_cols = xcffib.List(unpacker, 'B', self.length * 4)
        self.bufsize = unpacker.offset - base