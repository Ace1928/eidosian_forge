import xcffib
import struct
import io
def SetViewPort(self, screen, x, y, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2xII', screen, x, y))
    return self.send_request(12, buf, is_checked=is_checked)