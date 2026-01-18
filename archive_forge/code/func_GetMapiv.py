import xcffib
import struct
import io
from . import xproto
def GetMapiv(self, context_tag, target, query, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, target, query))
    return self.send_request(122, buf, GetMapivCookie, is_checked=is_checked)