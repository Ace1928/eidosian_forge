import xcffib
import struct
import io
from . import xproto
class GetPixelMapuivCookie(xcffib.Cookie):
    reply_type = GetPixelMapuivReply