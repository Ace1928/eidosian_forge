import xcffib
import struct
import io
from . import xproto
from . import render
class GetProviderPropertyCookie(xcffib.Cookie):
    reply_type = GetProviderPropertyReply