import xcffib
import struct
import io
from . import xproto
from . import xfixes
def CreateRegionFromBorderClip(self, region, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', region, window))
    return self.send_request(5, buf, is_checked=is_checked)