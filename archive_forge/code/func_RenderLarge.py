import xcffib
import struct
import io
from . import xproto
def RenderLarge(self, context_tag, request_num, request_total, data_len, data, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHHI', context_tag, request_num, request_total, data_len))
    buf.write(xcffib.pack_list(data, 'B'))
    return self.send_request(2, buf, is_checked=is_checked)