import xcffib
import struct
import io
from . import xproto
def GetError(self, context_tag, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context_tag))
    return self.send_request(115, buf, GetErrorCookie, is_checked=is_checked)