import xcffib
import struct
import io
from . import xproto
def Trapezoids(self, op, src, dst, mask_format, src_x, src_y, traps_len, traps, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3xIIIhh', op, src, dst, mask_format, src_x, src_y))
    buf.write(xcffib.pack_list(traps, TRAPEZOID))
    return self.send_request(10, buf, is_checked=is_checked)