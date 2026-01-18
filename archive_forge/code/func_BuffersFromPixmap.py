import xcffib
import struct
import io
from . import xproto
def BuffersFromPixmap(self, pixmap, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', pixmap))
    return self.send_request(8, buf, is_checked=is_checked)