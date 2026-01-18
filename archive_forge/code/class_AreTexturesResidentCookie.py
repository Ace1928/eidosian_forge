import xcffib
import struct
import io
from . import xproto
class AreTexturesResidentCookie(xcffib.Cookie):
    reply_type = AreTexturesResidentReply