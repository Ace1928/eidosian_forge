import xcffib
import struct
import io
from . import xproto
from . import render
class GetCrtcInfoCookie(xcffib.Cookie):
    reply_type = GetCrtcInfoReply