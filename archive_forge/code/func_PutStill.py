import xcffib
import struct
import io
from . import xproto
from . import shm
def PutStill(self, port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIhhHHhhHH', port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h))
    return self.send_request(6, buf, is_checked=is_checked)