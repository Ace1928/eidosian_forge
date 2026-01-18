import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIGrabDevice(self, window, time, cursor, deviceid, mode, paired_device_mode, owner_events, mask_len, mask, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIHBBBxH', window, time, cursor, deviceid, mode, paired_device_mode, owner_events, mask_len))
    buf.write(xcffib.pack_list(mask, 'I'))
    return self.send_request(51, buf, XIGrabDeviceCookie, is_checked=is_checked)