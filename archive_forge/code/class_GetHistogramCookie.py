import xcffib
import struct
import io
from . import xproto
class GetHistogramCookie(xcffib.Cookie):
    reply_type = GetHistogramReply