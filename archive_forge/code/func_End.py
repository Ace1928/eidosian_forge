import xcffib
import struct
import io
def End(self, cmap, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', cmap))
    return self.send_request(2, buf, EndCookie, is_checked=is_checked)