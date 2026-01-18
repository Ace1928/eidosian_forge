import xcffib
import struct
import io
def ListFontsWithInfo(self, max_names, pattern_len, pattern, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', max_names, pattern_len))
    buf.write(xcffib.pack_list(pattern, 'c'))
    return self.send_request(50, buf, ListFontsWithInfoCookie, is_checked=is_checked)