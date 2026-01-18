import xcffib
import struct
import io
from . import xproto
from . import render
def GetCrtcTransform(self, crtc, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', crtc))
    return self.send_request(27, buf, GetCrtcTransformCookie, is_checked=is_checked)