import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetDeviceKeyMapping(self, device_id, first_keycode, count, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBBBx', device_id, first_keycode, count))
    return self.send_request(24, buf, GetDeviceKeyMappingCookie, is_checked=is_checked)