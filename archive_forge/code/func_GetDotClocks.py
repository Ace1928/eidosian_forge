import xcffib
import struct
import io
def GetDotClocks(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', screen))
    return self.send_request(13, buf, GetDotClocksCookie, is_checked=is_checked)