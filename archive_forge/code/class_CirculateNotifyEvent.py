import xcffib
import struct
import io
class CirculateNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.event, self.window, self.place = unpacker.unpack('xx2xII4xB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 26))
        buf.write(struct.pack('=x2xII4xB3x', self.event, self.window, self.place))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, event, window, place):
        self = cls.__new__(cls)
        self.event = event
        self.window = window
        self.place = place
        return self