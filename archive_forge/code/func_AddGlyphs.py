import xcffib
import struct
import io
from . import xproto
def AddGlyphs(self, glyphset, glyphs_len, glyphids, glyphs, data_len, data, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', glyphset, glyphs_len))
    buf.write(xcffib.pack_list(glyphids, 'I'))
    buf.write(xcffib.pack_list(glyphs, GLYPHINFO))
    buf.write(xcffib.pack_list(data, 'B'))
    return self.send_request(20, buf, is_checked=is_checked)