import xcffib
import struct
import io
def PolyFillArc(self, drawable, gc, arcs_len, arcs, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, gc))
    buf.write(xcffib.pack_list(arcs, ARC))
    return self.send_request(71, buf, is_checked=is_checked)