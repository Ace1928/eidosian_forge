import xcffib
import struct
import io
from . import xproto
def DestroyPbuffer(self, pbuffer, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', pbuffer))
    return self.send_request(28, buf, is_checked=is_checked)