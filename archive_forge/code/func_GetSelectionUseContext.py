import xcffib
import struct
import io
from . import xproto
def GetSelectionUseContext(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(18, buf, GetSelectionUseContextCookie, is_checked=is_checked)