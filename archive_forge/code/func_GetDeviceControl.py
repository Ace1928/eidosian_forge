import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetDeviceControl(self, control_id, device_id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHBx', control_id, device_id))
    return self.send_request(34, buf, GetDeviceControlCookie, is_checked=is_checked)