import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceControlCookie(xcffib.Cookie):
    reply_type = GetDeviceControlReply