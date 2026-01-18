import xcffib
import struct
import io
from . import xproto
class QueryAlarmCookie(xcffib.Cookie):
    reply_type = QueryAlarmReply