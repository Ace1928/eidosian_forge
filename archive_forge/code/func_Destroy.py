import xcffib
import struct
import io
from . import xproto
from . import xfixes
def Destroy(self, damage, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', damage))
    return self.send_request(2, buf, is_checked=is_checked)