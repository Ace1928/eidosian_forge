import xcffib
import struct
import io
from . import xproto
class QueryServerStringCookie(xcffib.Cookie):
    reply_type = QueryServerStringReply