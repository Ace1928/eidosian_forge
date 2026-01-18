import xcffib
import struct
import io
from . import xproto
def AddTraps(self, picture, x_off, y_off, traps_len, traps, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIhh', picture, x_off, y_off))
    buf.write(xcffib.pack_list(traps, TRAP))
    return self.send_request(32, buf, is_checked=is_checked)