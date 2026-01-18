import xcffib
import struct
import io
from . import xfixes
from . import xproto
class BellFeedbackState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len, self.percent, self.pitch, self.duration = unpacker.unpack('BBHB3xHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBHB3xHH', self.class_id, self.feedback_id, self.len, self.percent, self.pitch, self.duration))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, percent, pitch, duration):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.percent = percent
        self.pitch = pitch
        self.duration = duration
        return self