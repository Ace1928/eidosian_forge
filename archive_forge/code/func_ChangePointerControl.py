import xcffib
import struct
import io
def ChangePointerControl(self, acceleration_numerator, acceleration_denominator, threshold, do_acceleration, do_threshold, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xhhhBB', acceleration_numerator, acceleration_denominator, threshold, do_acceleration, do_threshold))
    return self.send_request(105, buf, is_checked=is_checked)