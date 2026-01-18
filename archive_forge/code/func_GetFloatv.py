import xcffib
import struct
import io
from . import xproto
def GetFloatv(self, context_tag, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, pname))
    return self.send_request(116, buf, GetFloatvCookie, is_checked=is_checked)