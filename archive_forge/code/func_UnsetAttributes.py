import xcffib
import struct
import io
from . import xproto
def UnsetAttributes(self, drawable, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', drawable))
    return self.send_request(4, buf, is_checked=is_checked)