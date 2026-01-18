import xcffib
import struct
import io
from . import xfixes
from . import xproto
class FeedbackState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len = unpacker.unpack('BBH')
        if self.class_id == FeedbackClass.Keyboard:
            self.pitch, self.duration, self.led_mask, self.led_values, self.global_auto_repeat, self.click, self.percent = unpacker.unpack('HHIIBBBx')
            self.auto_repeats = xcffib.List(unpacker, 'B', 32)
        if self.class_id == FeedbackClass.Pointer:
            self.accel_num, self.accel_denom, self.threshold = unpacker.unpack('2xHHH')
        if self.class_id == FeedbackClass.String:
            self.max_symbols, self.num_keysyms = unpacker.unpack('HH')
            self.keysyms = xcffib.List(unpacker, 'I', self.num_keysyms)
        if self.class_id == FeedbackClass.Integer:
            self.resolution, self.min_value, self.max_value = unpacker.unpack('Iii')
        if self.class_id == FeedbackClass.Led:
            self.led_mask, self.led_values = unpacker.unpack('II')
        if self.class_id == FeedbackClass.Bell:
            self.percent, self.pitch, self.duration = unpacker.unpack('B3xHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBH', self.class_id, self.feedback_id, self.len))
        if self.class_id & FeedbackClass.Keyboard:
            self.pitch = self.data.pop(0)
            self.duration = self.data.pop(0)
            self.led_mask = self.data.pop(0)
            self.led_values = self.data.pop(0)
            self.global_auto_repeat = self.data.pop(0)
            self.click = self.data.pop(0)
            self.percent = self.data.pop(0)
            self.auto_repeats = self.data.pop(0)
            buf.write(struct.pack('=HHIIBBBx', self.pitch, self.duration, self.led_mask, self.led_values, self.global_auto_repeat, self.click, self.percent))
            buf.write(xcffib.pack_list(self.auto_repeats, 'B'))
        if self.class_id & FeedbackClass.Pointer:
            self.accel_num = self.data.pop(0)
            self.accel_denom = self.data.pop(0)
            self.threshold = self.data.pop(0)
            buf.write(struct.pack('=2xHHH', self.accel_num, self.accel_denom, self.threshold))
        if self.class_id & FeedbackClass.String:
            self.max_symbols = self.data.pop(0)
            self.num_keysyms = self.data.pop(0)
            self.keysyms = self.data.pop(0)
            buf.write(struct.pack('=HH', self.max_symbols, self.num_keysyms))
            buf.write(xcffib.pack_list(self.keysyms, 'I'))
        if self.class_id & FeedbackClass.Integer:
            self.resolution = self.data.pop(0)
            self.min_value = self.data.pop(0)
            self.max_value = self.data.pop(0)
            buf.write(struct.pack('=Iii', self.resolution, self.min_value, self.max_value))
        if self.class_id & FeedbackClass.Led:
            self.led_mask = self.data.pop(0)
            self.led_values = self.data.pop(0)
            buf.write(struct.pack('=II', self.led_mask, self.led_values))
        if self.class_id & FeedbackClass.Bell:
            self.percent = self.data.pop(0)
            self.pitch = self.data.pop(0)
            self.duration = self.data.pop(0)
            buf.write(struct.pack('=B3xHH', self.percent, self.pitch, self.duration))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, data):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.data = data
        return self