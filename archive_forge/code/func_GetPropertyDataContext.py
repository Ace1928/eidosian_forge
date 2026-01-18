import xcffib
import struct
import io
from . import xproto
def GetPropertyDataContext(self, window, property, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, property))
    return self.send_request(13, buf, GetPropertyDataContextCookie, is_checked=is_checked)