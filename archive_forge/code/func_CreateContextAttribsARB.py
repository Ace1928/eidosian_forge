import xcffib
import struct
import io
from . import xproto
def CreateContextAttribsARB(self, context, fbconfig, screen, share_list, is_direct, num_attribs, attribs, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIB3xI', context, fbconfig, screen, share_list, is_direct, num_attribs))
    buf.write(xcffib.pack_list(attribs, 'I'))
    return self.send_request(34, buf, is_checked=is_checked)