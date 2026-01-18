import xcffib
import struct
import io
def AllowEvents(self, mode, time, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xI', mode, time))
    return self.send_request(35, buf, is_checked=is_checked)