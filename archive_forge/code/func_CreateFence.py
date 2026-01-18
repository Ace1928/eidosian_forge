import xcffib
import struct
import io
from . import xproto
def CreateFence(self, drawable, fence, initially_triggered, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIB', drawable, fence, initially_triggered))
    return self.send_request(14, buf, is_checked=is_checked)