import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIChangeHierarchy(self, num_changes, changes, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3x', num_changes))
    buf.write(xcffib.pack_list(changes, HierarchyChange))
    return self.send_request(43, buf, is_checked=is_checked)