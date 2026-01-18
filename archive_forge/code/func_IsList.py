import xcffib
import struct
import io
from . import xproto
def IsList(self, context_tag, list, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, list))
    return self.send_request(141, buf, IsListCookie, is_checked=is_checked)