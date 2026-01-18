import xcffib
import struct
import io
def PolyRectangle(self, drawable, gc, rectangles_len, rectangles, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, gc))
    buf.write(xcffib.pack_list(rectangles, RECTANGLE))
    return self.send_request(67, buf, is_checked=is_checked)