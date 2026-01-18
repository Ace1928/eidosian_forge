import xcffib
import struct
import io
def CloseFont(self, font, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', font))
    return self.send_request(46, buf, is_checked=is_checked)