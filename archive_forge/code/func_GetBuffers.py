import xcffib
import struct
import io
from . import xproto
def GetBuffers(self, drawable, count, attachments_len, attachments, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, count))
    buf.write(xcffib.pack_list(attachments, 'I'))
    return self.send_request(5, buf, GetBuffersCookie, is_checked=is_checked)