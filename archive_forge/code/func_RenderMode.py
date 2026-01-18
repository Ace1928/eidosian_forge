import xcffib
import struct
import io
from . import xproto
def RenderMode(self, context_tag, mode, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, mode))
    return self.send_request(107, buf, RenderModeCookie, is_checked=is_checked)