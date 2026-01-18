import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryAdaptorsCookie(xcffib.Cookie):
    reply_type = QueryAdaptorsReply