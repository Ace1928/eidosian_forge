import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIGetFocus(self, deviceid, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', deviceid))
    return self.send_request(50, buf, XIGetFocusCookie, is_checked=is_checked)