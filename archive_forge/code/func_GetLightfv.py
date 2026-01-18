import xcffib
import struct
import io
from . import xproto
def GetLightfv(self, context_tag, light, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, light, pname))
    return self.send_request(118, buf, GetLightfvCookie, is_checked=is_checked)