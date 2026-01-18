import xcffib
import struct
import io
from . import xproto
def NewList(self, context_tag, list, mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, list, mode))
    return self.send_request(101, buf, is_checked=is_checked)