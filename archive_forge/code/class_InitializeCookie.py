import xcffib
import struct
import io
from . import xproto
class InitializeCookie(xcffib.Cookie):
    reply_type = InitializeReply