import xcffib
import struct
import io
from . import xproto
def SetPictureFilter(self, picture, filter_len, filter, values_len, values, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', picture, filter_len))
    buf.write(xcffib.pack_list(filter, 'c'))
    buf.write(struct.pack('=4x'))
    buf.write(xcffib.pack_list(values, 'i'))
    return self.send_request(30, buf, is_checked=is_checked)