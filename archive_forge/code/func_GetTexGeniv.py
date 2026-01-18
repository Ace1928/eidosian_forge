import xcffib
import struct
import io
from . import xproto
def GetTexGeniv(self, context_tag, coord, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, coord, pname))
    return self.send_request(134, buf, GetTexGenivCookie, is_checked=is_checked)