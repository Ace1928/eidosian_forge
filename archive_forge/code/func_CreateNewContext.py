import xcffib
import struct
import io
from . import xproto
def CreateNewContext(self, context, fbconfig, screen, render_type, share_list, is_direct, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIIB3x', context, fbconfig, screen, render_type, share_list, is_direct))
    return self.send_request(24, buf, is_checked=is_checked)