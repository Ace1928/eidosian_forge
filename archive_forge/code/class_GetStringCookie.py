import xcffib
import struct
import io
from . import xproto
class GetStringCookie(xcffib.Cookie):
    reply_type = GetStringReply