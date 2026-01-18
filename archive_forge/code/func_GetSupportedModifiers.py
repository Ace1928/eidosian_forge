import xcffib
import struct
import io
from . import xproto
def GetSupportedModifiers(self, window, depth, bpp, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIBB2x', window, depth, bpp))
    return self.send_request(6, buf, is_checked=is_checked)