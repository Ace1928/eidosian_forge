import xcffib
import struct
import io
class PropertyNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.window, self.atom, self.time, self.state = unpacker.unpack('xx2xIIIB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 28))
        buf.write(struct.pack('=x2xIIIB3x', self.window, self.atom, self.time, self.state))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, window, atom, time, state):
        self = cls.__new__(cls)
        self.window = window
        self.atom = atom
        self.time = time
        self.state = state
        return self