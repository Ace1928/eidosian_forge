import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIGetPropertyCookie(xcffib.Cookie):
    reply_type = XIGetPropertyReply