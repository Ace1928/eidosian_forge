import xcffib
import struct
import io
from . import xproto
def Flush(self, context_tag, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context_tag))
    return self.send_request(142, buf, is_checked=is_checked)