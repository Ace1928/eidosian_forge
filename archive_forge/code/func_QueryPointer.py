import xcffib
import struct
import io
def QueryPointer(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(38, buf, QueryPointerCookie, is_checked=is_checked)