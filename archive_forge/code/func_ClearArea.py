import xcffib
import struct
import io
def ClearArea(self, exposures, window, x, y, width, height, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIhhHH', exposures, window, x, y, width, height))
    return self.send_request(61, buf, is_checked=is_checked)