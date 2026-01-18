import xcffib
import struct
import io
def ListFonts(self, max_names, pattern_len, pattern, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', max_names, pattern_len))
    buf.write(xcffib.pack_list(pattern, 'c'))
    return self.send_request(49, buf, ListFontsCookie, is_checked=is_checked)