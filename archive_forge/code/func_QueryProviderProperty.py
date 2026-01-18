import xcffib
import struct
import io
from . import xproto
from . import render
def QueryProviderProperty(self, provider, property, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', provider, property))
    return self.send_request(37, buf, QueryProviderPropertyCookie, is_checked=is_checked)