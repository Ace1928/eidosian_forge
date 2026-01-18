import xcffib
import struct
import io
class MapNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.event, self.window, self.override_redirect = unpacker.unpack('xx2xIIB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 19))
        buf.write(struct.pack('=x2xIIB3x', self.event, self.window, self.override_redirect))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, event, window, override_redirect):
        self = cls.__new__(cls)
        self.event = event
        self.window = window
        self.override_redirect = override_redirect
        return self