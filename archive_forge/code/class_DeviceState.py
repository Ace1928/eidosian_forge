import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.control_id, self.len = unpacker.unpack('HH')
        if self.control_id == DeviceControl.resolution:
            self.num_valuators, = unpacker.unpack('I')
            self.resolution_values = xcffib.List(unpacker, 'I', self.num_valuators)
            unpacker.pad('I')
            self.resolution_min = xcffib.List(unpacker, 'I', self.num_valuators)
            unpacker.pad('I')
            self.resolution_max = xcffib.List(unpacker, 'I', self.num_valuators)
        if self.control_id == DeviceControl.abs_calib:
            self.min_x, self.max_x, self.min_y, self.max_y, self.flip_x, self.flip_y, self.rotation, self.button_threshold = unpacker.unpack('iiiiIIII')
        if self.control_id == DeviceControl.core:
            self.status, self.iscore = unpacker.unpack('BB2x')
        if self.control_id == DeviceControl.enable:
            self.enable, = unpacker.unpack('B3x')
        if self.control_id == DeviceControl.abs_area:
            self.offset_x, self.offset_y, self.width, self.height, self.screen, self.following = unpacker.unpack('IIIIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HH', self.control_id, self.len))
        if self.control_id & DeviceControl.resolution:
            self.num_valuators = self.data.pop(0)
            self.resolution_values = self.data.pop(0)
            self.resolution_min = self.data.pop(0)
            self.resolution_max = self.data.pop(0)
            buf.write(struct.pack('=I', self.num_valuators))
            buf.write(xcffib.pack_list(self.resolution_values, 'I'))
            buf.write(xcffib.pack_list(self.resolution_min, 'I'))
            buf.write(xcffib.pack_list(self.resolution_max, 'I'))
        if self.control_id & DeviceControl.abs_calib:
            self.min_x = self.data.pop(0)
            self.max_x = self.data.pop(0)
            self.min_y = self.data.pop(0)
            self.max_y = self.data.pop(0)
            self.flip_x = self.data.pop(0)
            self.flip_y = self.data.pop(0)
            self.rotation = self.data.pop(0)
            self.button_threshold = self.data.pop(0)
            buf.write(struct.pack('=iiiiIIII', self.min_x, self.max_x, self.min_y, self.max_y, self.flip_x, self.flip_y, self.rotation, self.button_threshold))
        if self.control_id & DeviceControl.core:
            self.status = self.data.pop(0)
            self.iscore = self.data.pop(0)
            buf.write(struct.pack('=BB2x', self.status, self.iscore))
        if self.control_id & DeviceControl.enable:
            self.enable = self.data.pop(0)
            buf.write(struct.pack('=B3x', self.enable))
        if self.control_id & DeviceControl.abs_area:
            self.offset_x = self.data.pop(0)
            self.offset_y = self.data.pop(0)
            self.width = self.data.pop(0)
            self.height = self.data.pop(0)
            self.screen = self.data.pop(0)
            self.following = self.data.pop(0)
            buf.write(struct.pack('=IIIIII', self.offset_x, self.offset_y, self.width, self.height, self.screen, self.following))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, control_id, len, data):
        self = cls.__new__(cls)
        self.control_id = control_id
        self.len = len
        self.data = data
        return self