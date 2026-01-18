import xcffib
import struct
import io
def WarpPointer(self, src_window, dst_window, src_x, src_y, src_width, src_height, dst_x, dst_y, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIhhHHhh', src_window, dst_window, src_x, src_y, src_width, src_height, dst_x, dst_y))
    return self.send_request(41, buf, is_checked=is_checked)