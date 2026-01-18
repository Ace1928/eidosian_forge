import xcffib
import struct
import io
from . import xproto
class WaitSBCCookie(xcffib.Cookie):
    reply_type = WaitSBCReply