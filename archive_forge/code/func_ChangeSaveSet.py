import xcffib
import struct
import io
def ChangeSaveSet(self, mode, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xI', mode, window))
    return self.send_request(6, buf, is_checked=is_checked)