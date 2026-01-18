import xcffib
import struct
import io
def QueryVersion(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)