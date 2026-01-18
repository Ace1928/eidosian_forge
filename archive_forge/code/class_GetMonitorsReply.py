import xcffib
import struct
import io
from . import xproto
from . import render
class GetMonitorsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.nMonitors, self.nOutputs = unpacker.unpack('xx2x4xIII12x')
        self.monitors = xcffib.List(unpacker, MonitorInfo, self.nMonitors)
        self.bufsize = unpacker.offset - base