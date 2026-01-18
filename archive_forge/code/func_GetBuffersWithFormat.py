import xcffib
import struct
import io
from . import xproto
def GetBuffersWithFormat(self, drawable, count, attachments_len, attachments, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, count))
    buf.write(xcffib.pack_list(attachments, AttachFormat))
    return self.send_request(7, buf, GetBuffersWithFormatCookie, is_checked=is_checked)