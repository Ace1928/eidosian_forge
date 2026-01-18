import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GrabDeviceKey(self, grab_window, num_classes, modifiers, modifier_device, grabbed_device, key, this_device_mode, other_device_mode, owner_events, classes, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHHBBBBBB2x', grab_window, num_classes, modifiers, modifier_device, grabbed_device, key, this_device_mode, other_device_mode, owner_events))
    buf.write(xcffib.pack_list(classes, 'I'))
    return self.send_request(15, buf, is_checked=is_checked)