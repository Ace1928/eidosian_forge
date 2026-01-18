import xcffib
import struct
import io
class SelectionRequestEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.time, self.owner, self.requestor, self.selection, self.target, self.property = unpacker.unpack('xx2xIIIIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 30))
        buf.write(struct.pack('=x2xIIIIII', self.time, self.owner, self.requestor, self.selection, self.target, self.property))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, time, owner, requestor, selection, target, property):
        self = cls.__new__(cls)
        self.time = time
        self.owner = owner
        self.requestor = requestor
        self.selection = selection
        self.target = target
        self.property = property
        return self