import xcffib
import struct
import io
from . import xfixes
from . import xproto
def UngrabDevice(self, time, device_id, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', time, device_id))
    return self.send_request(14, buf, is_checked=is_checked)