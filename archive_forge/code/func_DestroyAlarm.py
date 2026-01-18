import xcffib
import struct
import io
from . import xproto
def DestroyAlarm(self, alarm, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', alarm))
    return self.send_request(11, buf, is_checked=is_checked)