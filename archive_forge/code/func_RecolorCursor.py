import xcffib
import struct
import io
def RecolorCursor(self, cursor, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHHHHHH', cursor, fore_red, fore_green, fore_blue, back_red, back_green, back_blue))
    return self.send_request(96, buf, is_checked=is_checked)