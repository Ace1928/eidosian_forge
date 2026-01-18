import xcffib
import struct
import io
from . import xfixes
from . import xproto
class IntegerFeedbackState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len, self.resolution, self.min_value, self.max_value = unpacker.unpack('BBHIii')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBHIii', self.class_id, self.feedback_id, self.len, self.resolution, self.min_value, self.max_value))
        return buf.getvalue()
    fixed_size = 16

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, resolution, min_value, max_value):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.resolution = resolution
        self.min_value = min_value
        self.max_value = max_value
        return self