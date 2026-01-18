import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryPortAttributesCookie(xcffib.Cookie):
    reply_type = QueryPortAttributesReply