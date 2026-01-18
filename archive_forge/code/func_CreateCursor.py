import xcffib
import struct
import io
def CreateCursor(self, cid, source, mask, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, x, y, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIHHHHHHHH', cid, source, mask, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, x, y))
    return self.send_request(93, buf, is_checked=is_checked)