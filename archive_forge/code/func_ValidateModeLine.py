import xcffib
import struct
import io
def ValidateModeLine(self, screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, private, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHHHHHHHHH2xI12xI', screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize))
    buf.write(xcffib.pack_list(private, 'B'))
    return self.send_request(9, buf, ValidateModeLineCookie, is_checked=is_checked)