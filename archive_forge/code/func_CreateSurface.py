import xcffib
import struct
import io
from . import xv
def CreateSurface(self, surface_id, context_id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', surface_id, context_id))
    return self.send_request(4, buf, CreateSurfaceCookie, is_checked=is_checked)