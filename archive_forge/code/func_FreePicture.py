import xcffib
import struct
import io
from . import xproto
def FreePicture(self, picture, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', picture))
    return self.send_request(7, buf, is_checked=is_checked)