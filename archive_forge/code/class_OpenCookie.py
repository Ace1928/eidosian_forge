import xcffib
import struct
import io
from . import xproto
class OpenCookie(xcffib.Cookie):
    reply_type = OpenReply