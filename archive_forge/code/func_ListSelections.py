import xcffib
import struct
import io
from . import xproto
def ListSelections(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(21, buf, ListSelectionsCookie, is_checked=is_checked)