import xcffib
import struct
import io
from . import xproto
from . import render
def GetCrtcInfo(self, crtc, config_timestamp, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', crtc, config_timestamp))
    return self.send_request(20, buf, GetCrtcInfoCookie, is_checked=is_checked)