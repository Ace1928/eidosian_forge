import xcffib
import struct
import io
from . import xfixes
from . import xproto
def ListDeviceProperties(self, device_id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3x', device_id))
    return self.send_request(36, buf, ListDevicePropertiesCookie, is_checked=is_checked)