import xcffib
import struct
import io
from . import xproto
from . import render
def CreateLease(self, window, lid, num_crtcs, num_outputs, crtcs, outputs, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHH', window, lid, num_crtcs, num_outputs))
    buf.write(xcffib.pack_list(crtcs, 'I'))
    buf.write(xcffib.pack_list(outputs, 'I'))
    return self.send_request(45, buf, CreateLeaseCookie, is_checked=is_checked)