import xcffib
import struct
import io
def ModModeLine(self, screen, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, private, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHHHHHHHHH2xI12xI', screen, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize))
    buf.write(xcffib.pack_list(private, 'B'))
    return self.send_request(2, buf, is_checked=is_checked)