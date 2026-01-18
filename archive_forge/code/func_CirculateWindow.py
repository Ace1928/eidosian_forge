import xcffib
import struct
import io
def CirculateWindow(self, direction, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xI', direction, window))
    return self.send_request(13, buf, is_checked=is_checked)