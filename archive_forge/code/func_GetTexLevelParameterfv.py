import xcffib
import struct
import io
from . import xproto
def GetTexLevelParameterfv(self, context_tag, target, level, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIiI', context_tag, target, level, pname))
    return self.send_request(138, buf, GetTexLevelParameterfvCookie, is_checked=is_checked)