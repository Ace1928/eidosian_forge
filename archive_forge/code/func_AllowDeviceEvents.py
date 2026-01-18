import xcffib
import struct
import io
from . import xfixes
from . import xproto
def AllowDeviceEvents(self, time, mode, device_id, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIBB2x', time, mode, device_id))
    return self.send_request(19, buf, is_checked=is_checked)