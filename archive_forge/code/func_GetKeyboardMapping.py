import xcffib
import struct
import io
def GetKeyboardMapping(self, first_keycode, count, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBB', first_keycode, count))
    return self.send_request(101, buf, GetKeyboardMappingCookie, is_checked=is_checked)