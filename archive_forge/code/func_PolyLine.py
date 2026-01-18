import xcffib
import struct
import io
def PolyLine(self, coordinate_mode, drawable, gc, points_len, points, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xII', coordinate_mode, drawable, gc))
    buf.write(xcffib.pack_list(points, POINT))
    return self.send_request(65, buf, is_checked=is_checked)