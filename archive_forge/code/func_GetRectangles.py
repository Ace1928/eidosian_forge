import xcffib
import struct
import io
from . import xproto
def GetRectangles(self, window, source_kind, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', window, source_kind))
    return self.send_request(8, buf, GetRectanglesCookie, is_checked=is_checked)