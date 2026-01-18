import xcffib
import struct
import io
def CreateGlyphCursor(self, cid, source_font, mask_font, source_char, mask_char, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIHHHHHHHH', cid, source_font, mask_font, source_char, mask_char, fore_red, fore_green, fore_blue, back_red, back_green, back_blue))
    return self.send_request(94, buf, is_checked=is_checked)