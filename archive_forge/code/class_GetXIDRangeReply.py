import xcffib
import struct
import io
class GetXIDRangeReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.start_id, self.count = unpacker.unpack('xx2x4xII')
        self.bufsize = unpacker.offset - base