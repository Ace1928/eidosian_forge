import xcffib
import struct
import io
from . import xproto
class GetLightfvCookie(xcffib.Cookie):
    reply_type = GetLightfvReply