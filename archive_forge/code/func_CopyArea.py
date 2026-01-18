import xcffib
import struct
import io
def CopyArea(self, src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIhhhhHH', src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height))
    return self.send_request(62, buf, is_checked=is_checked)