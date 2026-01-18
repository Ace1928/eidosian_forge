import xcffib
import struct
import io
from . import xv
class CreateSubpictureReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.width_actual, self.height_actual, self.num_palette_entries, self.entry_bytes = unpacker.unpack('xx2x4xHHHH')
        self.component_order = xcffib.List(unpacker, 'B', 4)
        unpacker.unpack('12x')
        unpacker.pad('I')
        self.priv_data = xcffib.List(unpacker, 'I', self.length)
        self.bufsize = unpacker.offset - base