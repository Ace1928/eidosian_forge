import xcffib
import struct
import io
from . import xfixes
from . import xproto
def DeviceBell(self, device_id, feedback_id, feedback_class, percent, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBBBb', device_id, feedback_id, feedback_class, percent))
    return self.send_request(32, buf, is_checked=is_checked)