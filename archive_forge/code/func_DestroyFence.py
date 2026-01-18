import xcffib
import struct
import io
from . import xproto
def DestroyFence(self, fence, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', fence))
    return self.send_request(17, buf, is_checked=is_checked)