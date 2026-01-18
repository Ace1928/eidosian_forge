import xcffib
import struct
import io
def SetClipRectangles(self, ordering, gc, clip_x_origin, clip_y_origin, rectangles_len, rectangles, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIhh', ordering, gc, clip_x_origin, clip_y_origin))
    buf.write(xcffib.pack_list(rectangles, RECTANGLE))
    return self.send_request(59, buf, is_checked=is_checked)