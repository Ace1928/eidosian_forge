import xcffib
import struct
import io
from . import xproto
def GetLightiv(self, context_tag, light, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, light, pname))
    return self.send_request(119, buf, GetLightivCookie, is_checked=is_checked)