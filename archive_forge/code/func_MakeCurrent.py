import xcffib
import struct
import io
from . import xproto
def MakeCurrent(self, drawable, context, old_context_tag, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', drawable, context, old_context_tag))
    return self.send_request(5, buf, MakeCurrentCookie, is_checked=is_checked)