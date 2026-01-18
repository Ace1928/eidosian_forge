import xcffib
import struct
import io
from . import xproto
def QueryPictIndexValues(self, format, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', format))
    return self.send_request(2, buf, QueryPictIndexValuesCookie, is_checked=is_checked)