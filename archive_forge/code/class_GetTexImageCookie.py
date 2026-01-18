import xcffib
import struct
import io
from . import xproto
class GetTexImageCookie(xcffib.Cookie):
    reply_type = GetTexImageReply