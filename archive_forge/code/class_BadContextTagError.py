import xcffib
import struct
import io
from . import xproto
class BadContextTagError(xcffib.Error):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Error.__init__(self, unpacker)
        base = unpacker.offset
        self.bad_value, self.minor_opcode, self.major_opcode = unpacker.unpack('xx2xIHB21x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 4))
        buf.write(struct.pack('=x2xIHB21x', self.bad_value, self.minor_opcode, self.major_opcode))
        return buf.getvalue()