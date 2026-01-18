import xcffib
import struct
import io
from . import xproto
from . import render
def GetPanning(self, crtc, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', crtc))
    return self.send_request(28, buf, GetPanningCookie, is_checked=is_checked)