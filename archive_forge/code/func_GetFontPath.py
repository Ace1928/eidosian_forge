import xcffib
import struct
import io
def GetFontPath(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(52, buf, GetFontPathCookie, is_checked=is_checked)