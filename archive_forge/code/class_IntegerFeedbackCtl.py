import xcffib
import struct
import io
from . import xfixes
from . import xproto
class IntegerFeedbackCtl(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len, self.int_to_display = unpacker.unpack('BBHi')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBHi', self.class_id, self.feedback_id, self.len, self.int_to_display))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, int_to_display):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.int_to_display = int_to_display
        return self