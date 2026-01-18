import xcffib
import struct
import io
def QueryColors(self, cmap, pixels_len, pixels, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', cmap))
    buf.write(xcffib.pack_list(pixels, 'I'))
    return self.send_request(91, buf, QueryColorsCookie, is_checked=is_checked)