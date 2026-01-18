import xcffib
import struct
import io
from . import xproto
class IsEnabledCookie(xcffib.Cookie):
    reply_type = IsEnabledReply