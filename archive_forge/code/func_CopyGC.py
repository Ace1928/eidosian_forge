import xcffib
import struct
import io
def CopyGC(self, src_gc, dst_gc, value_mask, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', src_gc, dst_gc, value_mask))
    return self.send_request(57, buf, is_checked=is_checked)