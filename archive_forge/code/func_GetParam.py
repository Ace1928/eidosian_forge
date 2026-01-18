import xcffib
import struct
import io
from . import xproto
def GetParam(self, drawable, param, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, param))
    return self.send_request(13, buf, GetParamCookie, is_checked=is_checked)