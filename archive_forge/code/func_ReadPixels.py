import xcffib
import struct
import io
from . import xproto
def ReadPixels(self, context_tag, x, y, width, height, format, type, swap_bytes, lsb_first, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIiiiiIIBB', context_tag, x, y, width, height, format, type, swap_bytes, lsb_first))
    return self.send_request(111, buf, ReadPixelsCookie, is_checked=is_checked)