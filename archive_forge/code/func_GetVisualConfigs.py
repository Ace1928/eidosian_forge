import xcffib
import struct
import io
from . import xproto
def GetVisualConfigs(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', screen))
    return self.send_request(14, buf, GetVisualConfigsCookie, is_checked=is_checked)