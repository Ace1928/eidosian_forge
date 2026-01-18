import xcffib
import struct
import io
from . import xproto
def ForceLevel(self, power_level, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH', power_level))
    return self.send_request(6, buf, is_checked=is_checked)