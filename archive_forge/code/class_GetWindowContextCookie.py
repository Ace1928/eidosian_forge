import xcffib
import struct
import io
from . import xproto
class GetWindowContextCookie(xcffib.Cookie):
    reply_type = GetWindowContextReply