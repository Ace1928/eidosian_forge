import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetExtensionVersion(self, name_len, name, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', name_len))
    buf.write(xcffib.pack_list(name, 'c'))
    return self.send_request(1, buf, GetExtensionVersionCookie, is_checked=is_checked)