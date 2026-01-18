import xcffib
import struct
import io
from . import xfixes
from . import xproto
class RawKeyReleaseEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.detail, self.sourceid, self.valuators_len, self.flags = unpacker.unpack('xx2xHIIHHI4x')
        self.valuator_mask = xcffib.List(unpacker, 'I', self.valuators_len)
        unpacker.pad(FP3232)
        self.axisvalues = xcffib.List(unpacker, FP3232, sum(self.valuator_mask))
        unpacker.pad(FP3232)
        self.axisvalues_raw = xcffib.List(unpacker, FP3232, sum(self.valuator_mask))
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 14))
        buf.write(struct.pack('=x2xHIIHHI4x', self.deviceid, self.time, self.detail, self.sourceid, self.valuators_len, self.flags))
        buf.write(xcffib.pack_list(self.valuator_mask, 'I'))
        buf.write(xcffib.pack_list(self.axisvalues, FP3232))
        buf.write(xcffib.pack_list(self.axisvalues_raw, FP3232))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, detail, sourceid, valuators_len, flags, valuator_mask, axisvalues, axisvalues_raw):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.detail = detail
        self.sourceid = sourceid
        self.valuators_len = valuators_len
        self.flags = flags
        self.valuator_mask = valuator_mask
        self.axisvalues = axisvalues
        self.axisvalues_raw = axisvalues_raw
        return self