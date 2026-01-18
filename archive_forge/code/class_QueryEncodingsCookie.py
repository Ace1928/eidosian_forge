import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryEncodingsCookie(xcffib.Cookie):
    reply_type = QueryEncodingsReply