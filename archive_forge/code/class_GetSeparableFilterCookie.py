import xcffib
import struct
import io
from . import xproto
class GetSeparableFilterCookie(xcffib.Cookie):
    reply_type = GetSeparableFilterReply