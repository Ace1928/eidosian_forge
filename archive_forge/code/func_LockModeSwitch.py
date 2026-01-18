import xcffib
import struct
import io
def LockModeSwitch(self, screen, lock, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', screen, lock))
    return self.send_request(5, buf, is_checked=is_checked)