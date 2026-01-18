import xcffib
import struct
import io
def QueryDirectRenderingCapable(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', screen))
    return self.send_request(1, buf, QueryDirectRenderingCapableCookie, is_checked=is_checked)