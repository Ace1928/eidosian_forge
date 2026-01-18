import xcffib
import struct
import io
def CopyColormapAndFree(self, mid, src_cmap, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', mid, src_cmap))
    return self.send_request(80, buf, is_checked=is_checked)