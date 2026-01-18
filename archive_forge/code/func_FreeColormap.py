import xcffib
import struct
import io
def FreeColormap(self, cmap, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', cmap))
    return self.send_request(79, buf, is_checked=is_checked)