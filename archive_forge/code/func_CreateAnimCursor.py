import xcffib
import struct
import io
from . import xproto
def CreateAnimCursor(self, cid, cursors_len, cursors, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', cid))
    buf.write(xcffib.pack_list(cursors, ANIMCURSORELT))
    return self.send_request(31, buf, is_checked=is_checked)