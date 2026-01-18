import xcffib
import struct
import io
from . import xproto
class AuthenticateCookie(xcffib.Cookie):
    reply_type = AuthenticateReply