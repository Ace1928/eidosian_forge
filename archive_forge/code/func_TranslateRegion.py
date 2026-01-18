import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def TranslateRegion(self, region, dx, dy, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIhh', region, dx, dy))
    return self.send_request(17, buf, is_checked=is_checked)