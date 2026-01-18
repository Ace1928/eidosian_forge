import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def SetPictureClipRegion(self, picture, region, x_origin, y_origin, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIhh', picture, region, x_origin, y_origin))
    return self.send_request(22, buf, is_checked=is_checked)