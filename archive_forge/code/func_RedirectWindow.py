import xcffib
import struct
import io
from . import xproto
from . import xfixes
def RedirectWindow(self, window, update, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', window, update))
    return self.send_request(1, buf, is_checked=is_checked)