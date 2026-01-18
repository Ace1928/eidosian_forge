import xcffib
import struct
import io
from . import xproto
def GetSelectionCreateContext(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(16, buf, GetSelectionCreateContextCookie, is_checked=is_checked)