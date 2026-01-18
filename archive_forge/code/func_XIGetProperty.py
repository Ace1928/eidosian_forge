import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIGetProperty(self, deviceid, delete, property, type, offset, len, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHBxIIII', deviceid, delete, property, type, offset, len))
    return self.send_request(59, buf, XIGetPropertyCookie, is_checked=is_checked)