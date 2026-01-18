import xcffib
import struct
import io
def GetWindowAttributes(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(3, buf, GetWindowAttributesCookie, is_checked=is_checked)