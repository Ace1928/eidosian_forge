import xcffib
import struct
import io
def StoreNamedColor(self, flags, cmap, pixel, name_len, name, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIIH2x', flags, cmap, pixel, name_len))
    buf.write(xcffib.pack_list(name, 'c'))
    return self.send_request(90, buf, is_checked=is_checked)