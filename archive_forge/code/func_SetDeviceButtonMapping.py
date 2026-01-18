import xcffib
import struct
import io
from . import xfixes
from . import xproto
def SetDeviceButtonMapping(self, device_id, map_size, map, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBB2x', device_id, map_size))
    buf.write(xcffib.pack_list(map, 'B'))
    return self.send_request(29, buf, SetDeviceButtonMappingCookie, is_checked=is_checked)