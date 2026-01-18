import xcffib
import struct
import io
from . import xfixes
from . import xproto
def SetDeviceValuators(self, device_id, first_valuator, num_valuators, valuators, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBBBx', device_id, first_valuator, num_valuators))
    buf.write(xcffib.pack_list(valuators, 'i'))
    return self.send_request(33, buf, SetDeviceValuatorsCookie, is_checked=is_checked)