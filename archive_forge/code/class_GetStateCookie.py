import xcffib
import struct
import io
from . import xproto
class GetStateCookie(xcffib.Cookie):
    reply_type = GetStateReply