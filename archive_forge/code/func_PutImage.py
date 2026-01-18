import xcffib
import struct
import io
def PutImage(self, format, drawable, gc, width, height, dst_x, dst_y, left_pad, depth, data_len, data, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIIHHhhBB2x', format, drawable, gc, width, height, dst_x, dst_y, left_pad, depth))
    buf.write(xcffib.pack_list(data, 'B'))
    return self.send_request(72, buf, is_checked=is_checked)