import xcffib
import struct
import io
def GetMotionEvents(self, window, start, stop, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', window, start, stop))
    return self.send_request(39, buf, GetMotionEventsCookie, is_checked=is_checked)