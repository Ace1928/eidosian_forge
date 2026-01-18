import xcffib
import struct
import io
def UngrabPointer(self, time, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', time))
    return self.send_request(27, buf, is_checked=is_checked)