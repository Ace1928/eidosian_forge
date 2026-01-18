import xcffib
import struct
import io
from . import xproto
class InfoNotifyEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.power_level, self.state = unpacker.unpack('xx2x2xIHB21x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=x2x2xIHB21x', self.timestamp, self.power_level, self.state))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, timestamp, power_level, state):
        self = cls.__new__(cls)
        self.timestamp = timestamp
        self.power_level = power_level
        self.state = state
        return self