import xcffib
import struct
import io
from . import xproto
def ReferenceGlyphSet(self, gsid, existing, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', gsid, existing))
    return self.send_request(18, buf, is_checked=is_checked)