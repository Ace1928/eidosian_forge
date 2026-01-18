import xcffib
import struct
import io
def QueryKeymap(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(44, buf, QueryKeymapCookie, is_checked=is_checked)