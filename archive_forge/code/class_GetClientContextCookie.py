import xcffib
import struct
import io
from . import xproto
class GetClientContextCookie(xcffib.Cookie):
    reply_type = GetClientContextReply