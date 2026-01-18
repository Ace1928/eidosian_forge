import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceButtonStateNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.device_id, = unpacker.unpack('xB2x')
        self.buttons = xcffib.List(unpacker, 'B', 28)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 14))
        buf.write(struct.pack('=B2x', self.device_id))
        buf.write(xcffib.pack_list(self.buttons, 'B'))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, device_id, buttons):
        self = cls.__new__(cls)
        self.device_id = device_id
        self.buttons = buttons
        return self