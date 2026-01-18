import xcffib
import struct
import io
from . import xproto
from . import render
class GetPanningCookie(xcffib.Cookie):
    reply_type = GetPanningReply