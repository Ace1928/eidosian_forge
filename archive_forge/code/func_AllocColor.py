import xcffib
import struct
import io
def AllocColor(self, cmap, red, green, blue, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHHH2x', cmap, red, green, blue))
    return self.send_request(84, buf, AllocColorCookie, is_checked=is_checked)