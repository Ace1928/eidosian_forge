import xcffib
import struct
import io
from . import xproto
class RenderModeReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.ret_val, self.n, self.new_mode = unpacker.unpack('xx2x4xIII12x')
        self.data = xcffib.List(unpacker, 'I', self.n)
        self.bufsize = unpacker.offset - base