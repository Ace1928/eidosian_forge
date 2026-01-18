import xcffib
import struct
import io
from . import xproto
def CreateSolidFill(self, picture, color, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', picture))
    buf.write(color.pack() if hasattr(color, 'pack') else COLOR.synthetic(*color).pack())
    return self.send_request(33, buf, is_checked=is_checked)