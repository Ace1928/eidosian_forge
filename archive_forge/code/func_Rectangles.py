import xcffib
import struct
import io
from . import xproto
def Rectangles(self, operation, destination_kind, ordering, destination_window, x_offset, y_offset, rectangles_len, rectangles, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBBBxIhh', operation, destination_kind, ordering, destination_window, x_offset, y_offset))
    buf.write(xcffib.pack_list(rectangles, xproto.RECTANGLE))
    return self.send_request(1, buf, is_checked=is_checked)