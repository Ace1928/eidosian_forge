import xcffib
import struct
import io
def GetContext(self, context, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context))
    return self.send_request(4, buf, GetContextCookie, is_checked=is_checked)