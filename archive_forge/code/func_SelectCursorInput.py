import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def SelectCursorInput(self, window, event_mask, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, event_mask))
    return self.send_request(3, buf, is_checked=is_checked)