import xcffib
import struct
import io
from . import xproto
class CounterNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.kind, self.counter = unpacker.unpack('xB2xI')
        self.wait_value = INT64(unpacker)
        unpacker.pad(INT64)
        self.counter_value = INT64(unpacker)
        self.timestamp, self.count, self.destroyed = unpacker.unpack('IHBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=B2xI', self.kind, self.counter))
        buf.write(self.wait_value.pack() if hasattr(self.wait_value, 'pack') else INT64.synthetic(*self.wait_value).pack())
        buf.write(self.counter_value.pack() if hasattr(self.counter_value, 'pack') else INT64.synthetic(*self.counter_value).pack())
        buf.write(struct.pack('=I', self.timestamp))
        buf.write(struct.pack('=H', self.count))
        buf.write(struct.pack('=B', self.destroyed))
        buf.write(struct.pack('=x'))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, kind, counter, wait_value, counter_value, timestamp, count, destroyed):
        self = cls.__new__(cls)
        self.kind = kind
        self.counter = counter
        self.wait_value = wait_value
        self.counter_value = counter_value
        self.timestamp = timestamp
        self.count = count
        self.destroyed = destroyed
        return self