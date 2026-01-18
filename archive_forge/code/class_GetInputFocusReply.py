import xcffib
import struct
import io
class GetInputFocusReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.revert_to, self.focus = unpacker.unpack('xB2x4xI')
        self.bufsize = unpacker.offset - base