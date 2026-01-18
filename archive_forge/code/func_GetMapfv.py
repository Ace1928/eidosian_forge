import xcffib
import struct
import io
from . import xproto
def GetMapfv(self, context_tag, target, query, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, target, query))
    return self.send_request(121, buf, GetMapfvCookie, is_checked=is_checked)