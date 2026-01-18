import xcffib
import struct
import io
from . import xfixes
from . import xproto
def CloseDevice(self, device_id, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3x', device_id))
    return self.send_request(4, buf, is_checked=is_checked)