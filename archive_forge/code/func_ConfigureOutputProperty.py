import xcffib
import struct
import io
from . import xproto
from . import render
def ConfigureOutputProperty(self, output, property, pending, range, values_len, values, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIBB2x', output, property, pending, range))
    buf.write(xcffib.pack_list(values, 'i'))
    return self.send_request(12, buf, is_checked=is_checked)