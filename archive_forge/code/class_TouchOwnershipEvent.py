import xcffib
import struct
import io
from . import xfixes
from . import xproto
class TouchOwnershipEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.touchid, self.root, self.event, self.child, self.sourceid, self.flags = unpacker.unpack('xx2xHIIIIIH2xI8x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 21))
        buf.write(struct.pack('=x2xHIIIIIH2xI8x', self.deviceid, self.time, self.touchid, self.root, self.event, self.child, self.sourceid, self.flags))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, touchid, root, event, child, sourceid, flags):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.touchid = touchid
        self.root = root
        self.event = event
        self.child = child
        self.sourceid = sourceid
        self.flags = flags
        return self