import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceModifierMappingCookie(xcffib.Cookie):
    reply_type = GetDeviceModifierMappingReply