import xcffib
import struct
import io
from . import xproto
def GetMSC(self, drawable, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', drawable))
    return self.send_request(9, buf, GetMSCCookie, is_checked=is_checked)