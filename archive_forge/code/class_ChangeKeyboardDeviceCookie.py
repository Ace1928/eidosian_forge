import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ChangeKeyboardDeviceCookie(xcffib.Cookie):
    reply_type = ChangeKeyboardDeviceReply