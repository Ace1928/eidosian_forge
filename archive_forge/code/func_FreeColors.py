import xcffib
import struct
import io
def FreeColors(self, cmap, plane_mask, pixels_len, pixels, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', cmap, plane_mask))
    buf.write(xcffib.pack_list(pixels, 'I'))
    return self.send_request(88, buf, is_checked=is_checked)