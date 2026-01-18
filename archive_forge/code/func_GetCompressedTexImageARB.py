import xcffib
import struct
import io
from . import xproto
def GetCompressedTexImageARB(self, context_tag, target, level, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIi', context_tag, target, level))
    return self.send_request(160, buf, GetCompressedTexImageARBCookie, is_checked=is_checked)