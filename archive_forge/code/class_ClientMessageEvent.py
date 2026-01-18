import xcffib
import struct
import io
class ClientMessageEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.format, self.window, self.type = unpacker.unpack('xB2xII')
        self.data = ClientMessageData(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 33))
        buf.write(struct.pack('=B2xII', self.format, self.window, self.type))
        buf.write(self.data.pack() if hasattr(self.data, 'pack') else ClientMessageData.synthetic(*self.data).pack())
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, format, window, type, data):
        self = cls.__new__(cls)
        self.format = format
        self.window = window
        self.type = type
        self.data = data
        return self