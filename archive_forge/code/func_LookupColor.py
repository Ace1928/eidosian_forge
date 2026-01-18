import xcffib
import struct
import io
def LookupColor(self, cmap, name_len, name, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', cmap, name_len))
    buf.write(xcffib.pack_list(name, 'c'))
    return self.send_request(92, buf, LookupColorCookie, is_checked=is_checked)