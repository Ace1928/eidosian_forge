import xcffib
import struct
import io
from . import xproto
def QueryClientIds(self, num_specs, specs, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', num_specs))
    buf.write(xcffib.pack_list(specs, ClientIdSpec))
    return self.send_request(4, buf, QueryClientIdsCookie, is_checked=is_checked)