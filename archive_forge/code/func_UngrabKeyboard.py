import xcffib
import struct
import io
def UngrabKeyboard(self, time, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', time))
    return self.send_request(32, buf, is_checked=is_checked)