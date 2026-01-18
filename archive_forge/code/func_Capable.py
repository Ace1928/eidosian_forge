import xcffib
import struct
import io
from . import xproto
def Capable(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(1, buf, CapableCookie, is_checked=is_checked)