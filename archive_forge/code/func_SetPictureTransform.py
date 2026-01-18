import xcffib
import struct
import io
from . import xproto
def SetPictureTransform(self, picture, transform, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', picture))
    buf.write(transform.pack() if hasattr(transform, 'pack') else TRANSFORM.synthetic(*transform).pack())
    return self.send_request(28, buf, is_checked=is_checked)