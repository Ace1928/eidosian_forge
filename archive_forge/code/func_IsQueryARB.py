import xcffib
import struct
import io
from . import xproto
def IsQueryARB(self, context_tag, id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, id))
    return self.send_request(163, buf, IsQueryARBCookie, is_checked=is_checked)