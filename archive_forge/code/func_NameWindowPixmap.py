import xcffib
import struct
import io
from . import xproto
from . import xfixes
def NameWindowPixmap(self, window, pixmap, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, pixmap))
    return self.send_request(6, buf, is_checked=is_checked)