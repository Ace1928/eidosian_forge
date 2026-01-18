import xcffib
import struct
import io
from . import xproto
from . import render
def ChangeOutputProperty(self, output, property, type, format, mode, num_units, data, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIBB2xI', output, property, type, format, mode, num_units))
    buf.write(xcffib.pack_list(data, 'c'))
    return self.send_request(13, buf, is_checked=is_checked)