import xcffib
import struct
import io
class SelectionClearEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.time, self.owner, self.selection = unpacker.unpack('xx2xIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 29))
        buf.write(struct.pack('=x2xIII', self.time, self.owner, self.selection))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, time, owner, selection):
        self = cls.__new__(cls)
        self.time = time
        self.owner = owner
        self.selection = selection
        return self