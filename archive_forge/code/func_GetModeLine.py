import xcffib
import struct
import io
def GetModeLine(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', screen))
    return self.send_request(1, buf, GetModeLineCookie, is_checked=is_checked)