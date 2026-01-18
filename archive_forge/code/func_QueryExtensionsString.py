import xcffib
import struct
import io
from . import xproto
def QueryExtensionsString(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', screen))
    return self.send_request(18, buf, QueryExtensionsStringCookie, is_checked=is_checked)