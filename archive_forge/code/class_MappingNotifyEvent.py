import xcffib
import struct
import io
class MappingNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.request, self.first_keycode, self.count = unpacker.unpack('xx2xBBBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 34))
        buf.write(struct.pack('=x2xBBBx', self.request, self.first_keycode, self.count))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, request, first_keycode, count):
        self = cls.__new__(cls)
        self.request = request
        self.first_keycode = first_keycode
        self.count = count
        return self