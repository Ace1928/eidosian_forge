import xcffib
import struct
import io
from . import xproto
def GetBooleanv(self, context_tag, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', context_tag, pname))
    return self.send_request(112, buf, GetBooleanvCookie, is_checked=is_checked)