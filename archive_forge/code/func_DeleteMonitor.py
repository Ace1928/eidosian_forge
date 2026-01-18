import xcffib
import struct
import io
from . import xproto
from . import render
def DeleteMonitor(self, window, name, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, name))
    return self.send_request(44, buf, is_checked=is_checked)