import xcffib
import struct
import io
def ChangeKeyboardMapping(self, keycode_count, first_keycode, keysyms_per_keycode, keysyms, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xBB2x', keycode_count, first_keycode, keysyms_per_keycode))
    buf.write(xcffib.pack_list(keysyms, 'I'))
    return self.send_request(100, buf, is_checked=is_checked)