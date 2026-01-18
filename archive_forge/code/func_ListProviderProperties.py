import xcffib
import struct
import io
from . import xproto
from . import render
def ListProviderProperties(self, provider, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', provider))
    return self.send_request(36, buf, ListProviderPropertiesCookie, is_checked=is_checked)