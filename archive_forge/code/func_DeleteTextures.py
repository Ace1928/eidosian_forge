import xcffib
import struct
import io
from . import xproto
def DeleteTextures(self, context_tag, n, textures, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', context_tag, n))
    buf.write(xcffib.pack_list(textures, 'I'))
    return self.send_request(144, buf, is_checked=is_checked)