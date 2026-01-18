import xcffib
import struct
import io
def CopyPlane(self, src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height, bit_plane, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIhhhhHHI', src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height, bit_plane))
    return self.send_request(63, buf, is_checked=is_checked)