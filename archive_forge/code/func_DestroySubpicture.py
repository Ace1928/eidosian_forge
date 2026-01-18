import xcffib
import struct
import io
from . import xv
def DestroySubpicture(self, subpicture_id, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', subpicture_id))
    return self.send_request(7, buf, is_checked=is_checked)