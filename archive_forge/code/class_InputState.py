import xcffib
import struct
import io
from . import xfixes
from . import xproto
class InputState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len = unpacker.unpack('BB')
        if self.class_id == InputClass.Key:
            self.num_keys, = unpacker.unpack('Bx')
            self.keys = xcffib.List(unpacker, 'B', 32)
        if self.class_id == InputClass.Button:
            self.num_buttons, = unpacker.unpack('Bx')
            self.buttons = xcffib.List(unpacker, 'B', 32)
        if self.class_id == InputClass.Valuator:
            self.num_valuators, self.mode = unpacker.unpack('BB')
            self.valuators = xcffib.List(unpacker, 'i', self.num_valuators)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BB', self.class_id, self.len))
        if self.class_id & InputClass.Key:
            self.num_keys = self.data.pop(0)
            self.keys = self.data.pop(0)
            buf.write(struct.pack('=Bx', self.num_keys))
            buf.write(xcffib.pack_list(self.keys, 'B'))
        if self.class_id & InputClass.Button:
            self.num_buttons = self.data.pop(0)
            self.buttons = self.data.pop(0)
            buf.write(struct.pack('=Bx', self.num_buttons))
            buf.write(xcffib.pack_list(self.buttons, 'B'))
        if self.class_id & InputClass.Valuator:
            self.num_valuators = self.data.pop(0)
            self.mode = self.data.pop(0)
            self.valuators = self.data.pop(0)
            buf.write(struct.pack('=BB', self.num_valuators, self.mode))
            buf.write(xcffib.pack_list(self.valuators, 'i'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, class_id, len, data):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.data = data
        return self