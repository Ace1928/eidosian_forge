import xcffib
import struct
import io
def AddModeLine(self, screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, after_dotclock, after_hdisplay, after_hsyncstart, after_hsyncend, after_htotal, after_hskew, after_vdisplay, after_vsyncstart, after_vsyncend, after_vtotal, after_flags, private, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHHHHHHHHH2xI12xIIHHHHHHHHH2xI12x', screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, after_dotclock, after_hdisplay, after_hsyncstart, after_hsyncend, after_htotal, after_hskew, after_vdisplay, after_vsyncstart, after_vsyncend, after_vtotal, after_flags))
    buf.write(xcffib.pack_list(private, 'B'))
    return self.send_request(7, buf, is_checked=is_checked)