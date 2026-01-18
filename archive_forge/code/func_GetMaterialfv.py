import xcffib
import struct
import io
from . import xproto
def GetMaterialfv(self, context_tag, face, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, face, pname))
    return self.send_request(123, buf, GetMaterialfvCookie, is_checked=is_checked)