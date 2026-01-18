import xcffib
import struct
import io
from . import xproto
def CreateGLXPixmap(self, screen, visual, pixmap, glx_pixmap, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIII', screen, visual, pixmap, glx_pixmap))
    return self.send_request(13, buf, is_checked=is_checked)