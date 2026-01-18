import xcffib
import struct
import io
from . import xproto
def CreateGlyphSet(self, gsid, format, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', gsid, format))
    return self.send_request(17, buf, is_checked=is_checked)