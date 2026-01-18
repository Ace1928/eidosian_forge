import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def DestroyRegion(self, region, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', region))
    return self.send_request(10, buf, is_checked=is_checked)