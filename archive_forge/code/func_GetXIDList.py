import xcffib
import struct
import io
def GetXIDList(self, count, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', count))
    return self.send_request(2, buf, GetXIDListCookie, is_checked=is_checked)