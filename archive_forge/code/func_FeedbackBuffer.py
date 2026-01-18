import xcffib
import struct
import io
from . import xproto
def FeedbackBuffer(self, context_tag, size, type, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIii', context_tag, size, type))
    return self.send_request(105, buf, is_checked=is_checked)