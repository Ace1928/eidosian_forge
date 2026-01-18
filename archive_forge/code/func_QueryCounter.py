import xcffib
import struct
import io
from . import xproto
def QueryCounter(self, counter, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', counter))
    return self.send_request(5, buf, QueryCounterCookie, is_checked=is_checked)