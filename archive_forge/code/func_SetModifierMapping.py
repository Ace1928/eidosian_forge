import xcffib
import struct
import io
def SetModifierMapping(self, keycodes_per_modifier, keycodes, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2x', keycodes_per_modifier))
    buf.write(xcffib.pack_list(keycodes, 'B'))
    return self.send_request(118, buf, SetModifierMappingCookie, is_checked=is_checked)