import xcffib
import struct
import io
class bigreqExtension(xcffib.Extension):

    def Enable(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(0, buf, EnableCookie, is_checked=is_checked)