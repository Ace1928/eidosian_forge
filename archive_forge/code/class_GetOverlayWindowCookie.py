import xcffib
import struct
import io
from . import xproto
from . import xfixes
class GetOverlayWindowCookie(xcffib.Cookie):
    reply_type = GetOverlayWindowReply