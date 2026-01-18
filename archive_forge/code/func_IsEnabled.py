import xcffib
import struct
import io
from . import xproto
def IsEnabled(self, context_tag, capability, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, capability))
    return self.send_request(140, buf, IsEnabledCookie, is_checked=is_checked)