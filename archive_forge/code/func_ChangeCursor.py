import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def ChangeCursor(self, source, destination, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', source, destination))
    return self.send_request(26, buf, is_checked=is_checked)