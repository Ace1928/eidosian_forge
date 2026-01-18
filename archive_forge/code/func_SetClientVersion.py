import xcffib
import struct
import io
def SetClientVersion(self, major, minor, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', major, minor))
    return self.send_request(14, buf, is_checked=is_checked)