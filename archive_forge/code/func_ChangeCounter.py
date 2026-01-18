import xcffib
import struct
import io
from . import xproto
def ChangeCounter(self, counter, amount, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', counter))
    buf.write(amount.pack() if hasattr(amount, 'pack') else INT64.synthetic(*amount).pack())
    return self.send_request(4, buf, is_checked=is_checked)