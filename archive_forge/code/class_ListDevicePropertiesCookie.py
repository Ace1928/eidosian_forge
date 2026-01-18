import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ListDevicePropertiesCookie(xcffib.Cookie):
    reply_type = ListDevicePropertiesReply