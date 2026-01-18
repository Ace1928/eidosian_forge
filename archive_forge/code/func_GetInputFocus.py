import xcffib
import struct
import io
def GetInputFocus(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(43, buf, GetInputFocusCookie, is_checked=is_checked)