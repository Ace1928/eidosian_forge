import xcffib
import struct
import io
from . import xproto
from . import render
def GetMonitors(self, window, get_active, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB', window, get_active))
    return self.send_request(42, buf, GetMonitorsCookie, is_checked=is_checked)