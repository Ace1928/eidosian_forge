import xcffib
import struct
import io
from . import xproto
def GetTexEnvfv(self, context_tag, target, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
    return self.send_request(130, buf, GetTexEnvfvCookie, is_checked=is_checked)