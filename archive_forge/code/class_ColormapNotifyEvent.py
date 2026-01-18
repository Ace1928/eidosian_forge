import xcffib
import struct
import io
class ColormapNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.window, self.colormap, self.new, self.state = unpacker.unpack('xx2xIIBB2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 32))
        buf.write(struct.pack('=x2xIIBB2x', self.window, self.colormap, self.new, self.state))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, window, colormap, new, state):
        self = cls.__new__(cls)
        self.window = window
        self.colormap = colormap
        self.new = new
        self.state = state
        return self