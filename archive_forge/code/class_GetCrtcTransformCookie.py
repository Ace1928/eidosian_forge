import xcffib
import struct
import io
from . import xproto
from . import render
class GetCrtcTransformCookie(xcffib.Cookie):
    reply_type = GetCrtcTransformReply