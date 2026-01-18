import xcffib
import struct
import io
from . import xproto
class PictOpError(xcffib.Error):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Error.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 2))
        buf.write(struct.pack('=x2x'))
        return buf.getvalue()