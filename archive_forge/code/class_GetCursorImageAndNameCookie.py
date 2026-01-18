import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class GetCursorImageAndNameCookie(xcffib.Cookie):
    reply_type = GetCursorImageAndNameReply