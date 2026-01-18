import xcffib
import struct
import io
from . import xproto
def Triangles(self, op, src, dst, mask_format, src_x, src_y, triangles_len, triangles, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3xIIIhh', op, src, dst, mask_format, src_x, src_y))
    buf.write(xcffib.pack_list(triangles, TRIANGLE))
    return self.send_request(11, buf, is_checked=is_checked)