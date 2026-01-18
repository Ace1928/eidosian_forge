import xcffib
import struct
import io
from . import xproto
def BufferFromPixmap(self, pixmap, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', pixmap))
    return self.send_request(3, buf, BufferFromPixmapCookie, is_checked=is_checked)