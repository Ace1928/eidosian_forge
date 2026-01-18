import xcffib
import struct
import io
from . import xproto
from . import shm
def QueryImageAttributes(self, port, id, width, height, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHH', port, id, width, height))
    return self.send_request(17, buf, QueryImageAttributesCookie, is_checked=is_checked)