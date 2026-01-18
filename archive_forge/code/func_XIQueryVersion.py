import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIQueryVersion(self, major_version, minor_version, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', major_version, minor_version))
    return self.send_request(47, buf, XIQueryVersionCookie, is_checked=is_checked)