import xcffib
import struct
import io
from . import xproto
class QueryClientIdsCookie(xcffib.Cookie):
    reply_type = QueryClientIdsReply