import xcffib
import struct
import io
from . import xproto
from . import render
def SetScreenConfig(self, window, timestamp, config_timestamp, sizeID, rotation, rate, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIHHH2x', window, timestamp, config_timestamp, sizeID, rotation, rate))
    return self.send_request(2, buf, SetScreenConfigCookie, is_checked=is_checked)