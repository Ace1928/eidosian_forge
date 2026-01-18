import xcffib
import struct
import io
from . import xproto
class QueryScreensCookie(xcffib.Cookie):
    reply_type = QueryScreensReply