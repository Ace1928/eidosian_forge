import xcffib
import struct
import io
from . import xproto
class GetParamCookie(xcffib.Cookie):
    reply_type = GetParamReply