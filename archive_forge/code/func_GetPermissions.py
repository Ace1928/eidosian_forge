import xcffib
import struct
import io
def GetPermissions(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', screen))
    return self.send_request(20, buf, GetPermissionsCookie, is_checked=is_checked)