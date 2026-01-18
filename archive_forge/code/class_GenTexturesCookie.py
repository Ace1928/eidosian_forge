import xcffib
import struct
import io
from . import xproto
class GenTexturesCookie(xcffib.Cookie):
    reply_type = GenTexturesReply