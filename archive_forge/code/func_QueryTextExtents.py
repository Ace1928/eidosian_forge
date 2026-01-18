import xcffib
import struct
import io
def QueryTextExtents(self, font, string_len, string, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    buf.write(struct.pack('=B', string_len & 1))
    buf.write(struct.pack('=I', font))
    buf.write(xcffib.pack_list(string, CHAR2B))
    return self.send_request(48, buf, QueryTextExtentsCookie, is_checked=is_checked)