import xcffib
import struct
import io
from . import xproto
def GetScreenSize(self, window, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, screen))
    return self.send_request(3, buf, GetScreenSizeCookie, is_checked=is_checked)