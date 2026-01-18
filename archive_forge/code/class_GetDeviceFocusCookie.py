import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceFocusCookie(xcffib.Cookie):
    reply_type = GetDeviceFocusReply