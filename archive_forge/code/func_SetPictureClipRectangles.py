import xcffib
import struct
import io
from . import xproto
def SetPictureClipRectangles(self, picture, clip_x_origin, clip_y_origin, rectangles_len, rectangles, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIhh', picture, clip_x_origin, clip_y_origin))
    buf.write(xcffib.pack_list(rectangles, xproto.RECTANGLE))
    return self.send_request(6, buf, is_checked=is_checked)