import xcffib
import struct
import io
def DestroyContext(self, screen, context, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', screen, context))
    return self.send_request(6, buf, is_checked=is_checked)