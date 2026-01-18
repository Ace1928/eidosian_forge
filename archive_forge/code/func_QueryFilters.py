import xcffib
import struct
import io
from . import xproto
def QueryFilters(self, drawable, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', drawable))
    return self.send_request(29, buf, QueryFiltersCookie, is_checked=is_checked)