import xcffib
import struct
import io
from . import xproto
class FinishCookie(xcffib.Cookie):
    reply_type = FinishReply