import xcffib
import struct
import io
from . import xproto
def CreatePbuffer(self, screen, fbconfig, pbuffer, num_attribs, attribs, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIII', screen, fbconfig, pbuffer, num_attribs))
    buf.write(xcffib.pack_list(attribs, 'I'))
    return self.send_request(27, buf, is_checked=is_checked)