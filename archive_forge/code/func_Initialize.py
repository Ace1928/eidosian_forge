import xcffib
import struct
import io
from . import xproto
def Initialize(self, desired_major_version, desired_minor_version, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBB', desired_major_version, desired_minor_version))
    return self.send_request(0, buf, InitializeCookie, is_checked=is_checked)