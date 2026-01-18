import xcffib
import struct
import io
from . import xproto
def CreateDrawable(self, drawable, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', drawable))
    return self.send_request(3, buf, is_checked=is_checked)