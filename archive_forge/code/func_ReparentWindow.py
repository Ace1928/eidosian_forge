import xcffib
import struct
import io
def ReparentWindow(self, window, parent, x, y, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIhh', window, parent, x, y))
    return self.send_request(7, buf, is_checked=is_checked)