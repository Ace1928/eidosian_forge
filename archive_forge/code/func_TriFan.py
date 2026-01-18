import xcffib
import struct
import io
from . import xproto
def TriFan(self, op, src, dst, mask_format, src_x, src_y, points_len, points, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3xIIIhh', op, src, dst, mask_format, src_x, src_y))
    buf.write(xcffib.pack_list(points, POINTFIX))
    return self.send_request(13, buf, is_checked=is_checked)