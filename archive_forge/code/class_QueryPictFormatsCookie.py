import xcffib
import struct
import io
from . import xproto
class QueryPictFormatsCookie(xcffib.Cookie):
    reply_type = QueryPictFormatsReply