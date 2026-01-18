import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XISelectEvents(self, window, num_mask, masks, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', window, num_mask))
    buf.write(xcffib.pack_list(masks, EventMask))
    return self.send_request(46, buf, is_checked=is_checked)