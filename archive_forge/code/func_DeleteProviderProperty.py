import xcffib
import struct
import io
from . import xproto
from . import render
def DeleteProviderProperty(self, provider, property, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', provider, property))
    return self.send_request(40, buf, is_checked=is_checked)