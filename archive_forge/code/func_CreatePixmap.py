import xcffib
import struct
import io
def CreatePixmap(self, depth, pid, drawable, width, height, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIIHH', depth, pid, drawable, width, height))
    return self.send_request(53, buf, is_checked=is_checked)