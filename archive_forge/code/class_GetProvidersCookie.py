import xcffib
import struct
import io
from . import xproto
from . import render
class GetProvidersCookie(xcffib.Cookie):
    reply_type = GetProvidersReply