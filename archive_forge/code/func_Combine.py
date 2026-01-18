import xcffib
import struct
import io
from . import xproto
def Combine(self, operation, destination_kind, source_kind, destination_window, x_offset, y_offset, source_window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBBBxIhhI', operation, destination_kind, source_kind, destination_window, x_offset, y_offset, source_window))
    return self.send_request(3, buf, is_checked=is_checked)