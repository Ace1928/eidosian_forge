import xcffib
import struct
import io
from . import xproto
def CreateConicalGradient(self, picture, center, angle, num_stops, stops, colors, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', picture))
    buf.write(center.pack() if hasattr(center, 'pack') else POINTFIX.synthetic(*center).pack())
    buf.write(struct.pack('=i', angle))
    buf.write(struct.pack('=I', num_stops))
    buf.write(xcffib.pack_list(stops, 'i'))
    buf.write(xcffib.pack_list(colors, COLOR))
    return self.send_request(36, buf, is_checked=is_checked)