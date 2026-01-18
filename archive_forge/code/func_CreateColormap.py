import xcffib
import struct
import io
def CreateColormap(self, alloc, mid, window, visual, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIII', alloc, mid, window, visual))
    return self.send_request(78, buf, is_checked=is_checked)