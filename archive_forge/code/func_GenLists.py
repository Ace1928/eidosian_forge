import xcffib
import struct
import io
from . import xproto
def GenLists(self, context_tag, range, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', context_tag, range))
    return self.send_request(104, buf, GenListsCookie, is_checked=is_checked)