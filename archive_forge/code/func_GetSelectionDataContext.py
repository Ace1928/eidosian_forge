import xcffib
import struct
import io
from . import xproto
def GetSelectionDataContext(self, selection, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', selection))
    return self.send_request(20, buf, GetSelectionDataContextCookie, is_checked=is_checked)