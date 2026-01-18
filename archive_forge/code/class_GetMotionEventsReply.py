import xcffib
import struct
import io
class GetMotionEventsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.events_len, = unpacker.unpack('xx2x4xI20x')
        self.events = xcffib.List(unpacker, TIMECOORD, self.events_len)
        self.bufsize = unpacker.offset - base