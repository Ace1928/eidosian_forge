import xcffib
import struct
import io
from . import xproto
class GetPriorityCookie(xcffib.Cookie):
    reply_type = GetPriorityReply