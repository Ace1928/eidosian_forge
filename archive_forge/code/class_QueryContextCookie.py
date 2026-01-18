import xcffib
import struct
import io
from . import xproto
class QueryContextCookie(xcffib.Cookie):
    reply_type = QueryContextReply