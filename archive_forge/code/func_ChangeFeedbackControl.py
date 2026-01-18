import xcffib
import struct
import io
from . import xfixes
from . import xproto
def ChangeFeedbackControl(self, mask, device_id, feedback_id, feedback, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIBB2x', mask, device_id, feedback_id))
    buf.write(feedback.pack() if hasattr(feedback, 'pack') else FeedbackCtl.synthetic(*feedback).pack())
    return self.send_request(23, buf, is_checked=is_checked)