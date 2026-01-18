import xcffib
import struct
import io
from . import xproto
from . import render
def SetCrtcConfig(self, crtc, timestamp, config_timestamp, x, y, mode, rotation, outputs_len, outputs, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIhhIH2x', crtc, timestamp, config_timestamp, x, y, mode, rotation))
    buf.write(xcffib.pack_list(outputs, 'I'))
    return self.send_request(21, buf, SetCrtcConfigCookie, is_checked=is_checked)