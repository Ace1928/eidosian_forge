import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def CreateRegionFromGC(self, region, gc, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', region, gc))
    return self.send_request(8, buf, is_checked=is_checked)