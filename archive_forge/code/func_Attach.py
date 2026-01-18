import xcffib
import struct
import io
from . import xproto
def Attach(self, shmseg, shmid, read_only, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIB3x', shmseg, shmid, read_only))
    return self.send_request(1, buf, is_checked=is_checked)