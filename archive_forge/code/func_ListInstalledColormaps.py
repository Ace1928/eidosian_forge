import xcffib
import struct
import io
def ListInstalledColormaps(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(83, buf, ListInstalledColormapsCookie, is_checked=is_checked)