import xcffib
import struct
import io
from . import xfixes
from . import xproto
def SendExtensionEvent(self, destination, device_id, propagate, num_classes, num_events, events, classes, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIBBHB3x', destination, device_id, propagate, num_classes, num_events))
    buf.write(xcffib.pack_list(events, EventForSend))
    buf.write(xcffib.pack_list(classes, 'I'))
    return self.send_request(31, buf, is_checked=is_checked)