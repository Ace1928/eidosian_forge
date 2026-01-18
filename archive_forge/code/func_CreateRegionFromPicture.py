import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def CreateRegionFromPicture(self, region, picture, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', region, picture))
    return self.send_request(9, buf, is_checked=is_checked)