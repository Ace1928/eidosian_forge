import xcffib
import struct
import io
def GetGammaRamp(self, screen, size, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', screen, size))
    return self.send_request(17, buf, GetGammaRampCookie, is_checked=is_checked)