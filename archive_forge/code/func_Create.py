import xcffib
import struct
import io
from . import xproto
from . import xfixes
def Create(self, damage, drawable, level, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIB3x', damage, drawable, level))
    return self.send_request(1, buf, is_checked=is_checked)