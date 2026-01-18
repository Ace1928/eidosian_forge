import xcffib
import struct
import io
from . import xproto
from . import shm
def SelectVideoNotify(self, drawable, onoff, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', drawable, onoff))
    return self.send_request(10, buf, is_checked=is_checked)