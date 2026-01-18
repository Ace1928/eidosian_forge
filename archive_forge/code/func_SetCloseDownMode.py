import xcffib
import struct
import io
def SetCloseDownMode(self, mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2x', mode))
    return self.send_request(112, buf, is_checked=is_checked)