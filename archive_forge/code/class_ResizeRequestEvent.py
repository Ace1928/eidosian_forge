import xcffib
import struct
import io
class ResizeRequestEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.window, self.width, self.height = unpacker.unpack('xx2xIHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 25))
        buf.write(struct.pack('=x2xIHH', self.window, self.width, self.height))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, window, width, height):
        self = cls.__new__(cls)
        self.window = window
        self.width = width
        self.height = height
        return self