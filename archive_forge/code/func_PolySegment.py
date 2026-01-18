import xcffib
import struct
import io
def PolySegment(self, drawable, gc, segments_len, segments, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, gc))
    buf.write(xcffib.pack_list(segments, SEGMENT))
    return self.send_request(66, buf, is_checked=is_checked)