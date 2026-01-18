import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def HideCursor(self, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(29, buf, is_checked=is_checked)