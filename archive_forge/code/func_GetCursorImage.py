import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def GetCursorImage(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(4, buf, GetCursorImageCookie, is_checked=is_checked)