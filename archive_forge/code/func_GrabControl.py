import xcffib
import struct
import io
from . import xproto
def GrabControl(self, impervious, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3x', impervious))
    return self.send_request(3, buf, is_checked=is_checked)