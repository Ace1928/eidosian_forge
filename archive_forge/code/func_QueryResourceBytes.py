import xcffib
import struct
import io
from . import xproto
def QueryResourceBytes(self, client, num_specs, specs, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', client, num_specs))
    buf.write(xcffib.pack_list(specs, ResourceIdSpec))
    return self.send_request(5, buf, QueryResourceBytesCookie, is_checked=is_checked)