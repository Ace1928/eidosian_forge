import xcffib
import struct
import io
def DeleteProperty(self, window, property, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, property))
    return self.send_request(19, buf, is_checked=is_checked)