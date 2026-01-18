import xcffib
import struct
import io
def ListProperties(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(21, buf, ListPropertiesCookie, is_checked=is_checked)