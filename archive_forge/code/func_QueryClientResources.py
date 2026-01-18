import xcffib
import struct
import io
from . import xproto
def QueryClientResources(self, xid, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', xid))
    return self.send_request(2, buf, QueryClientResourcesCookie, is_checked=is_checked)