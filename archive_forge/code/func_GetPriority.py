import xcffib
import struct
import io
from . import xproto
def GetPriority(self, id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', id))
    return self.send_request(13, buf, GetPriorityCookie, is_checked=is_checked)