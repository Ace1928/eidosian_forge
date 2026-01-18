import xcffib
import struct
import io
from . import xproto
def Authenticate(self, window, magic, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, magic))
    return self.send_request(2, buf, AuthenticateCookie, is_checked=is_checked)