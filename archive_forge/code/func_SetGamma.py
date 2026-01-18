import xcffib
import struct
import io
def SetGamma(self, screen, red, green, blue, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2xIII12x', screen, red, green, blue))
    return self.send_request(15, buf, is_checked=is_checked)