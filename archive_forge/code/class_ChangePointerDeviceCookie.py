import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ChangePointerDeviceCookie(xcffib.Cookie):
    reply_type = ChangePointerDeviceReply