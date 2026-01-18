import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIAllowEvents(self, time, deviceid, event_mode, touchid, grab_window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHBxII', time, deviceid, event_mode, touchid, grab_window))
    return self.send_request(53, buf, is_checked=is_checked)