import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def CreateRegionFromWindow(self, region, window, kind, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIB3x', region, window, kind))
    return self.send_request(7, buf, is_checked=is_checked)