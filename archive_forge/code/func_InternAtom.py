import xcffib
import struct
import io
def InternAtom(self, only_if_exists, name_len, name, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xH2x', only_if_exists, name_len))
    buf.write(xcffib.pack_list(name, 'c'))
    return self.send_request(16, buf, InternAtomCookie, is_checked=is_checked)