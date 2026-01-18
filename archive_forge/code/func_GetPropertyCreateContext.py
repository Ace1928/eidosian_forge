import xcffib
import struct
import io
from . import xproto
def GetPropertyCreateContext(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(9, buf, GetPropertyCreateContextCookie, is_checked=is_checked)