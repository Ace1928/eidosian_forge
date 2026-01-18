import xcffib
import struct
import io
def SetDashes(self, gc, dash_offset, dashes_len, dashes, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHH', gc, dash_offset, dashes_len))
    buf.write(xcffib.pack_list(dashes, 'B'))
    return self.send_request(58, buf, is_checked=is_checked)