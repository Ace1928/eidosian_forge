import xcffib
import struct
import io
from . import xfixes
from . import xproto
class LedFeedbackCtl(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len, self.led_mask, self.led_values = unpacker.unpack('BBHII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBHII', self.class_id, self.feedback_id, self.len, self.led_mask, self.led_values))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, led_mask, led_values):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.led_mask = led_mask
        self.led_values = led_values
        return self