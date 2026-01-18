import xcffib
import struct
import io
from . import xproto
class QueryAlarmReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x')
        self.trigger = TRIGGER(unpacker)
        unpacker.pad(INT64)
        self.delta = INT64(unpacker)
        self.events, self.state = unpacker.unpack('BB2x')
        self.bufsize = unpacker.offset - base