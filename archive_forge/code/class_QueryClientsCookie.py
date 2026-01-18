import xcffib
import struct
import io
from . import xproto
class QueryClientsCookie(xcffib.Cookie):
    reply_type = QueryClientsReply