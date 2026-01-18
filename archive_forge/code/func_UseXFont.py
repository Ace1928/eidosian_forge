import xcffib
import struct
import io
from . import xproto
def UseXFont(self, context_tag, font, first, count, list_base, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIII', context_tag, font, first, count, list_base))
    return self.send_request(12, buf, is_checked=is_checked)