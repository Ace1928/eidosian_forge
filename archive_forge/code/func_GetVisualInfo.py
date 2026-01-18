import xcffib
import struct
import io
from . import xproto
def GetVisualInfo(self, n_drawables, drawables, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', n_drawables))
    buf.write(xcffib.pack_list(drawables, 'I'))
    return self.send_request(6, buf, GetVisualInfoCookie, is_checked=is_checked)