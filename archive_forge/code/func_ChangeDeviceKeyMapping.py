import xcffib
import struct
import io
from . import xfixes
from . import xproto
def ChangeDeviceKeyMapping(self, device_id, first_keycode, keysyms_per_keycode, keycode_count, keysyms, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBBBB', device_id, first_keycode, keysyms_per_keycode, keycode_count))
    buf.write(xcffib.pack_list(keysyms, 'I'))
    return self.send_request(25, buf, is_checked=is_checked)