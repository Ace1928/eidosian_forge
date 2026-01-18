import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ListInputDevicesCookie(xcffib.Cookie):
    reply_type = ListInputDevicesReply