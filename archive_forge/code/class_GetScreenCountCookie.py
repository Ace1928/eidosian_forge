import xcffib
import struct
import io
from . import xproto
class GetScreenCountCookie(xcffib.Cookie):
    reply_type = GetScreenCountReply