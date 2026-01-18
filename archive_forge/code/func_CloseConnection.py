import xcffib
import struct
import io
def CloseConnection(self, screen, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', screen))
    return self.send_request(3, buf, is_checked=is_checked)