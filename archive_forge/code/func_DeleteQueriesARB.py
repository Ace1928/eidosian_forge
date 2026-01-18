import xcffib
import struct
import io
from . import xproto
def DeleteQueriesARB(self, context_tag, n, ids, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', context_tag, n))
    buf.write(xcffib.pack_list(ids, 'I'))
    return self.send_request(161, buf, is_checked=is_checked)