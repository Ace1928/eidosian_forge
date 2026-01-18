import xcffib
import struct
import io
from . import xproto
from . import render
def GetScreenResourcesCurrent(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(25, buf, GetScreenResourcesCurrentCookie, is_checked=is_checked)