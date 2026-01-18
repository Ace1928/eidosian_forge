import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceMotionEventsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.num_events, self.num_axes, self.device_mode = unpacker.unpack('xB2x4xIBB18x')
        self.events = xcffib.List(unpacker, xcffib.__DeviceTimeCoord_wrapper(DeviceTimeCoord, self.num_axes), self.num_events)
        self.bufsize = unpacker.offset - base