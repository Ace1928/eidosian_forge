import xcffib
import struct
import io
from . import xproto
def ListSystemCounters(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(1, buf, ListSystemCountersCookie, is_checked=is_checked)