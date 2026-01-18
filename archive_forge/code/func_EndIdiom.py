import xcffib
import struct
import io
from . import xproto
def EndIdiom(self, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(5, buf, is_checked=is_checked)