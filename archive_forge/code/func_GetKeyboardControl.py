import xcffib
import struct
import io
def GetKeyboardControl(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(103, buf, GetKeyboardControlCookie, is_checked=is_checked)