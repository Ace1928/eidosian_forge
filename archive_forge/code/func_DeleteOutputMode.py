import xcffib
import struct
import io
from . import xproto
from . import render
def DeleteOutputMode(self, output, mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', output, mode))
    return self.send_request(19, buf, is_checked=is_checked)