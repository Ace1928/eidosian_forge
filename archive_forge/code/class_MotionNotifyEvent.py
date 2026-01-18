import xcffib
import struct
import io
class MotionNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.detail, self.time, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.state, self.same_screen = unpacker.unpack('xB2xIIIIhhhhHBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 6))
        buf.write(struct.pack('=B2xIIIIhhhhHBx', self.detail, self.time, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.state, self.same_screen))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, detail, time, root, event, child, root_x, root_y, event_x, event_y, state, same_screen):
        self = cls.__new__(cls)
        self.detail = detail
        self.time = time
        self.root = root
        self.event = event
        self.child = child
        self.root_x = root_x
        self.root_y = root_y
        self.event_x = event_x
        self.event_y = event_y
        self.state = state
        self.same_screen = same_screen
        return self