import xcffib
import struct
import io
from . import xproto
def VendorPrivate(self, vendor_code, context_tag, data_len, data, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', vendor_code, context_tag))
    buf.write(xcffib.pack_list(data, 'B'))
    return self.send_request(16, buf, is_checked=is_checked)