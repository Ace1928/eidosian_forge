import xcffib
import struct
import io
def AllocColorCells(self, contiguous, cmap, colors, planes, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIHH', contiguous, cmap, colors, planes))
    return self.send_request(86, buf, AllocColorCellsCookie, is_checked=is_checked)