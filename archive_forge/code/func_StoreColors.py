import xcffib
import struct
import io
def StoreColors(self, cmap, items_len, items, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', cmap))
    buf.write(xcffib.pack_list(items, COLORITEM))
    return self.send_request(89, buf, is_checked=is_checked)