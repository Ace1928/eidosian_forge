import xcffib
import struct
import io
from . import xproto
from . import render
def GetScreenSizeRange(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(6, buf, GetScreenSizeRangeCookie, is_checked=is_checked)