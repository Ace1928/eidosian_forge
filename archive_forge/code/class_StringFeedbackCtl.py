import xcffib
import struct
import io
from . import xfixes
from . import xproto
class StringFeedbackCtl(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len, self.num_keysyms = unpacker.unpack('BBH2xH')
        self.keysyms = xcffib.List(unpacker, 'I', self.num_keysyms)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBH2xH', self.class_id, self.feedback_id, self.len, self.num_keysyms))
        buf.write(xcffib.pack_list(self.keysyms, 'I'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, num_keysyms, keysyms):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.num_keysyms = num_keysyms
        self.keysyms = keysyms
        return self