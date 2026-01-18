import xcffib
import struct
import io
def SwitchMode(self, screen, zoom, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', screen, zoom))
    return self.send_request(3, buf, is_checked=is_checked)