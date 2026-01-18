import xcffib
import struct
import io
from . import xproto
from . import xfixes
def Subtract(self, damage, repair, parts, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', damage, repair, parts))
    return self.send_request(3, buf, is_checked=is_checked)