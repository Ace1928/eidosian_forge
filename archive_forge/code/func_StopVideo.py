import xcffib
import struct
import io
from . import xproto
from . import shm
def StopVideo(self, port, drawable, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', port, drawable))
    return self.send_request(9, buf, is_checked=is_checked)