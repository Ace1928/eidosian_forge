import xcffib
import struct
import io
from . import xproto
def IsActive(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(4, buf, IsActiveCookie, is_checked=is_checked)