import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def GetClientDisconnectMode(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(34, buf, GetClientDisconnectModeCookie, is_checked=is_checked)