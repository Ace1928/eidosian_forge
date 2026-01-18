import xcffib
import struct
import io
from . import xproto
class GetErrorCookie(xcffib.Cookie):
    reply_type = GetErrorReply