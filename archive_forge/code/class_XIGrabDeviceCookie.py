import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIGrabDeviceCookie(xcffib.Cookie):
    reply_type = XIGrabDeviceReply