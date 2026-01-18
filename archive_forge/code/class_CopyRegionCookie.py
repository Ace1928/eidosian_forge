import xcffib
import struct
import io
from . import xproto
class CopyRegionCookie(xcffib.Cookie):
    reply_type = CopyRegionReply