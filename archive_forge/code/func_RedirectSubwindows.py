import xcffib
import struct
import io
from . import xproto
from . import xfixes
def RedirectSubwindows(self, window, update, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', window, update))
    return self.send_request(2, buf, is_checked=is_checked)