import xcffib
import struct
import io
from . import xproto
from . import render
class GetProviderInfoCookie(xcffib.Cookie):
    reply_type = GetProviderInfoReply