import xcffib
import struct
import io
from . import xproto
class GetSelectionUseContextCookie(xcffib.Cookie):
    reply_type = GetSelectionUseContextReply