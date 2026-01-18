import xcffib
import struct
import io
def ImageText8(self, string_len, drawable, gc, x, y, string, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIIhh', string_len, drawable, gc, x, y))
    buf.write(xcffib.pack_list(string, 'c'))
    return self.send_request(76, buf, is_checked=is_checked)