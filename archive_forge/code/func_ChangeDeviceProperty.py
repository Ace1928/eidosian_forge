import xcffib
import struct
import io
from . import xfixes
from . import xproto
def ChangeDeviceProperty(self, property, type, device_id, format, mode, num_items, items, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIBBBxI', property, type, device_id, format, mode, num_items))
    if format & PropertyFormat._8Bits:
        data8 = items.pop(0)
        items.pop(0)
        buf.write(xcffib.pack_list(data8, 'B'))
        buf.write(struct.pack('=4x'))
    if format & PropertyFormat._16Bits:
        data16 = items.pop(0)
        items.pop(0)
        buf.write(xcffib.pack_list(data16, 'H'))
        buf.write(struct.pack('=4x'))
    if format & PropertyFormat._32Bits:
        data32 = items.pop(0)
        buf.write(xcffib.pack_list(data32, 'I'))
    return self.send_request(37, buf, is_checked=is_checked)