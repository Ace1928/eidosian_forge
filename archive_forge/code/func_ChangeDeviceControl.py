import xcffib
import struct
import io
from . import xfixes
from . import xproto
def ChangeDeviceControl(self, control_id, device_id, control, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHBx', control_id, device_id))
    buf.write(control.pack() if hasattr(control, 'pack') else DeviceCtl.synthetic(*control).pack())
    return self.send_request(35, buf, ChangeDeviceControlCookie, is_checked=is_checked)