import xcffib
import struct
import io
def SetInputFocus(self, revert_to, focus, time, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xII', revert_to, focus, time))
    return self.send_request(42, buf, is_checked=is_checked)