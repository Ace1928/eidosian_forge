import xcffib
import struct
import io
from . import xfixes
from . import xproto
class OpenDeviceCookie(xcffib.Cookie):
    reply_type = OpenDeviceReply