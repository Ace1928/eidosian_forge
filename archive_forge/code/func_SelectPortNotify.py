import xcffib
import struct
import io
from . import xproto
from . import shm
def SelectPortNotify(self, port, onoff, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', port, onoff))
    return self.send_request(11, buf, is_checked=is_checked)