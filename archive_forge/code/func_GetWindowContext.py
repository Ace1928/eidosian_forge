import xcffib
import struct
import io
from . import xproto
def GetWindowContext(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(7, buf, GetWindowContextCookie, is_checked=is_checked)