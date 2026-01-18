import xcffib
import struct
import io
from . import xproto
def GetPixelMapfv(self, context_tag, map, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, map))
    return self.send_request(125, buf, GetPixelMapfvCookie, is_checked=is_checked)