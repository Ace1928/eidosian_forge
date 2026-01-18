import xcffib
import struct
import io
from . import xproto
def GetClientContext(self, resource, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', resource))
    return self.send_request(22, buf, GetClientContextCookie, is_checked=is_checked)