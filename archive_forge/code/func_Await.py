import xcffib
import struct
import io
from . import xproto
def Await(self, wait_list_len, wait_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    buf.write(xcffib.pack_list(wait_list, WAITCONDITION))
    return self.send_request(7, buf, is_checked=is_checked)