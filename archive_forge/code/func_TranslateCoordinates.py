import xcffib
import struct
import io
def TranslateCoordinates(self, src_window, dst_window, src_x, src_y, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIhh', src_window, dst_window, src_x, src_y))
    return self.send_request(40, buf, TranslateCoordinatesCookie, is_checked=is_checked)