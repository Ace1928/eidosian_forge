import xcffib
import struct
import io
from . import xproto
from . import shm
def UngrabPort(self, port, time, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', port, time))
    return self.send_request(4, buf, is_checked=is_checked)