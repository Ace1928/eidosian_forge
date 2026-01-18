import xcffib
import struct
import io
from . import xproto
from . import randr
from . import xfixes
from . import sync
def QueryCapabilities(self, target, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', target))
    return self.send_request(4, buf, QueryCapabilitiesCookie, is_checked=is_checked)