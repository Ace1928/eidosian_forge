import xcffib
import struct
import io
from . import xproto
def GetWindowCreateContext(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(6, buf, GetWindowCreateContextCookie, is_checked=is_checked)