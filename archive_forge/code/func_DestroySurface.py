import xcffib
import struct
import io
from . import xv
def DestroySurface(self, surface_id, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', surface_id))
    return self.send_request(5, buf, is_checked=is_checked)