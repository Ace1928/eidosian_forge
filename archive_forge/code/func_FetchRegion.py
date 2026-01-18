import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def FetchRegion(self, region, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', region))
    return self.send_request(19, buf, FetchRegionCookie, is_checked=is_checked)