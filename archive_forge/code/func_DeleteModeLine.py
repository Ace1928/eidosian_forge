import xcffib
import struct
import io
def DeleteModeLine(self, screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, private, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHHHHHHHHH2xI12xI', screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize))
    buf.write(xcffib.pack_list(private, 'B'))
    return self.send_request(8, buf, is_checked=is_checked)