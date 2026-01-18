import xcffib
import struct
import io
from . import xproto
def GetPixelMapuiv(self, context_tag, map, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, map))
    return self.send_request(126, buf, GetPixelMapuivCookie, is_checked=is_checked)