import xcffib
import struct
import io
from . import xproto
def GetVersion(self, client_major_version, client_minor_version, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', client_major_version, client_minor_version))
    return self.send_request(0, buf, GetVersionCookie, is_checked=is_checked)