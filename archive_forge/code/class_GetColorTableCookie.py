import xcffib
import struct
import io
from . import xproto
class GetColorTableCookie(xcffib.Cookie):
    reply_type = GetColorTableReply