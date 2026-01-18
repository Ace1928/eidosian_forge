import xcffib
import struct
import io
def SetScreenSaver(self, timeout, interval, prefer_blanking, allow_exposures, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xhhBB', timeout, interval, prefer_blanking, allow_exposures))
    return self.send_request(107, buf, is_checked=is_checked)