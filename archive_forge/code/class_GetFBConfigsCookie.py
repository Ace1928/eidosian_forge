import xcffib
import struct
import io
from . import xproto
class GetFBConfigsCookie(xcffib.Cookie):
    reply_type = GetFBConfigsReply