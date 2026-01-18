import xcffib
import struct
import io
from . import xproto
def GetMinmax(self, context_tag, target, format, type, swap_bytes, reset, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIBB', context_tag, target, format, type, swap_bytes, reset))
    return self.send_request(157, buf, GetMinmaxCookie, is_checked=is_checked)