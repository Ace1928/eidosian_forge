import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def ShowCursor(self, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(30, buf, is_checked=is_checked)