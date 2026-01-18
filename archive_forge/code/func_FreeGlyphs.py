import xcffib
import struct
import io
from . import xproto
def FreeGlyphs(self, glyphset, glyphs_len, glyphs, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', glyphset))
    buf.write(xcffib.pack_list(glyphs, 'I'))
    return self.send_request(22, buf, is_checked=is_checked)