import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def SetRegion(self, region, rectangles_len, rectangles, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', region))
    buf.write(xcffib.pack_list(rectangles, xproto.RECTANGLE))
    return self.send_request(11, buf, is_checked=is_checked)