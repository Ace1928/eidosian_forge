import xcffib
import struct
import io
def SetSelectionOwner(self, owner, selection, time, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', owner, selection, time))
    return self.send_request(22, buf, is_checked=is_checked)