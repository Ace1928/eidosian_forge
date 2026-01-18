import xcffib
import struct
import io
from . import xproto
class FDFromFenceCookie(xcffib.Cookie):
    reply_type = FDFromFenceReply