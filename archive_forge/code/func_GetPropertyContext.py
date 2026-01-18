import xcffib
import struct
import io
from . import xproto
def GetPropertyContext(self, window, property, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, property))
    return self.send_request(12, buf, GetPropertyContextCookie, is_checked=is_checked)