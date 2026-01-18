import xcffib
import struct
import io
from . import xproto
def SetPriority(self, id, priority, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', id, priority))
    return self.send_request(12, buf, is_checked=is_checked)