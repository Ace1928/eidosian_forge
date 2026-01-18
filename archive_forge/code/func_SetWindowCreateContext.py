import xcffib
import struct
import io
from . import xproto
def SetWindowCreateContext(self, context_len, context, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context_len))
    buf.write(xcffib.pack_list(context, 'c'))
    return self.send_request(5, buf, is_checked=is_checked)