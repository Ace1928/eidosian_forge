import xcffib
import struct
import io
def ForceScreenSaver(self, mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2x', mode))
    return self.send_request(115, buf, is_checked=is_checked)