import xcffib
import struct
import io
def UngrabServer(self, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(37, buf, is_checked=is_checked)