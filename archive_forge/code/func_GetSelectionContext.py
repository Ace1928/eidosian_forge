import xcffib
import struct
import io
from . import xproto
def GetSelectionContext(self, selection, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', selection))
    return self.send_request(19, buf, GetSelectionContextCookie, is_checked=is_checked)