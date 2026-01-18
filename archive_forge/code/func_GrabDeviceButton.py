import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GrabDeviceButton(self, grab_window, grabbed_device, modifier_device, num_classes, modifiers, this_device_mode, other_device_mode, button, owner_events, classes, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIBBHHBBBB2x', grab_window, grabbed_device, modifier_device, num_classes, modifiers, this_device_mode, other_device_mode, button, owner_events))
    buf.write(xcffib.pack_list(classes, 'I'))
    return self.send_request(17, buf, is_checked=is_checked)