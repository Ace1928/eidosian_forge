import xcffib
import struct
import io
from . import xproto
def SetCounter(self, counter, value, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', counter))
    buf.write(value.pack() if hasattr(value, 'pack') else INT64.synthetic(*value).pack())
    return self.send_request(3, buf, is_checked=is_checked)