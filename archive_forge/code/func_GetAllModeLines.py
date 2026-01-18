import xcffib
import struct
import io
def GetAllModeLines(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', screen))
    return self.send_request(6, buf, GetAllModeLinesCookie, is_checked=is_checked)