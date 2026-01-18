import xcffib
import struct
import io
from . import xproto
def FDFromFence(self, drawable, fence, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, fence))
    return self.send_request(5, buf, FDFromFenceCookie, is_checked=is_checked)