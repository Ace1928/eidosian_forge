import xcffib
import struct
import io
from . import xproto
from . import render
class OutputChange(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.config_timestamp, self.window, self.output, self.crtc, self.mode, self.rotation, self.connection, self.subpixel_order = unpacker.unpack('IIIIIIHBB')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IIIIIIHBB', self.timestamp, self.config_timestamp, self.window, self.output, self.crtc, self.mode, self.rotation, self.connection, self.subpixel_order))
        return buf.getvalue()
    fixed_size = 28

    @classmethod
    def synthetic(cls, timestamp, config_timestamp, window, output, crtc, mode, rotation, connection, subpixel_order):
        self = cls.__new__(cls)
        self.timestamp = timestamp
        self.config_timestamp = config_timestamp
        self.window = window
        self.output = output
        self.crtc = crtc
        self.mode = mode
        self.rotation = rotation
        self.connection = connection
        self.subpixel_order = subpixel_order
        return self