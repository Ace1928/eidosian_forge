import xcffib
import struct
import io
from . import xfixes
from . import xproto
class PtrFeedbackState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len, self.accel_num, self.accel_denom, self.threshold = unpacker.unpack('BBH2xHHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBH2xHHH', self.class_id, self.feedback_id, self.len, self.accel_num, self.accel_denom, self.threshold))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, accel_num, accel_denom, threshold):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.accel_num = accel_num
        self.accel_denom = accel_denom
        self.threshold = threshold
        return self