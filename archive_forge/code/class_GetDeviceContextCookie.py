import xcffib
import struct
import io
from . import xproto
class GetDeviceContextCookie(xcffib.Cookie):
    reply_type = GetDeviceContextReply