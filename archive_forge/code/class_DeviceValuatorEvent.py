import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceValuatorEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.device_id, self.device_state, self.num_valuators, self.first_valuator = unpacker.unpack('xB2xHBB')
        self.valuators = xcffib.List(unpacker, 'i', 6)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=B2xHBB', self.device_id, self.device_state, self.num_valuators, self.first_valuator))
        buf.write(xcffib.pack_list(self.valuators, 'i'))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, device_id, device_state, num_valuators, first_valuator, valuators):
        self = cls.__new__(cls)
        self.device_id = device_id
        self.device_state = device_state
        self.num_valuators = num_valuators
        self.first_valuator = first_valuator
        self.valuators = valuators
        return self