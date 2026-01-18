import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetDeviceMotionEvents(self, start, stop, device_id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIB3x', start, stop, device_id))
    return self.send_request(10, buf, GetDeviceMotionEventsCookie, is_checked=is_checked)