import xcffib
import struct
import io
from . import xproto
class BufferSwapCompleteEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.event_type, self.drawable, self.ust_hi, self.ust_lo, self.msc_hi, self.msc_lo, self.sbc = unpacker.unpack('xx2xH2xIIIIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=x2xH2xIIIIII', self.event_type, self.drawable, self.ust_hi, self.ust_lo, self.msc_hi, self.msc_lo, self.sbc))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, event_type, drawable, ust_hi, ust_lo, msc_hi, msc_lo, sbc):
        self = cls.__new__(cls)
        self.event_type = event_type
        self.drawable = drawable
        self.ust_hi = ust_hi
        self.ust_lo = ust_lo
        self.msc_hi = msc_hi
        self.msc_lo = msc_lo
        self.sbc = sbc
        return self