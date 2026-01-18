import xcffib
import struct
import io
def GetGamma(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH26x', screen))
    return self.send_request(16, buf, GetGammaCookie, is_checked=is_checked)